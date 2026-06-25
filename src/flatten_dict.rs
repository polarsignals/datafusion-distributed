//! Normalizes nested (dictionary-of-dictionary) columns across the Flight wire.
//!
//! Arrow IPC cannot represent a dictionary whose values are *themselves* a
//! dictionary: an IPC `Field` holds a single `DictionaryEncoding` slot and the
//! schema encoder recurses a dictionary straight to its leaf value type, so the
//! inner dictionary's encoding is dropped and the reader mis-reads the buffers
//! ("Buffer count mismatched with metadata").
//!
//! Such columns do occur in practice (e.g. symbolized profiling data carried as
//! `first_value` aggregate state: `Dictionary(UInt32, Dictionary(UInt32, Utf8))`).
//! To ship them we flatten each nested dictionary down to a single level before
//! encoding ([`flatten_schema`] + [`cast_batch_to_schema`]) and restore the
//! declared dict-of-dict shape on the receiving side, also via
//! [`cast_batch_to_schema`]. Both casts round-trip losslessly.

use std::sync::Arc;

use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, FieldRef, Fields, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatchOptions;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::{Stream, StreamExt};

/// Collapse any directly-nested dictionary layers to a single level, recursing
/// through container types. `Dictionary(K, Dictionary(.., leaf))` becomes
/// `Dictionary(K, leaf)`; all other types are returned structurally unchanged.
pub fn flatten_data_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Dictionary(key, value) => {
            // Flatten the value first (it may be a dict, or a container holding
            // dicts), then peel off any remaining dictionary layers so the
            // dictionary's values are never themselves a dictionary.
            let mut value = flatten_data_type(value);
            while let DataType::Dictionary(_, inner) = value {
                value = *inner;
            }
            DataType::Dictionary(key.clone(), Box::new(value))
        }
        DataType::Struct(fields) => DataType::Struct(flatten_fields(fields)),
        DataType::List(f) => DataType::List(flatten_field(f)),
        DataType::LargeList(f) => DataType::LargeList(flatten_field(f)),
        DataType::ListView(f) => DataType::ListView(flatten_field(f)),
        DataType::LargeListView(f) => DataType::LargeListView(flatten_field(f)),
        DataType::FixedSizeList(f, n) => DataType::FixedSizeList(flatten_field(f), *n),
        DataType::Map(f, sorted) => DataType::Map(flatten_field(f), *sorted),
        other => other.clone(),
    }
}

fn flatten_field(field: &FieldRef) -> FieldRef {
    let flat = flatten_data_type(field.data_type());
    if &flat == field.data_type() {
        return Arc::clone(field);
    }
    Arc::new(field.as_ref().clone().with_data_type(flat))
}

fn flatten_fields(fields: &Fields) -> Fields {
    fields.iter().map(flatten_field).collect()
}

/// Returns a flattened copy of `schema`, or `None` if it contains no nested
/// dictionaries (so callers can skip all work in the common case).
pub fn flatten_schema(schema: &SchemaRef) -> Option<SchemaRef> {
    let fields = flatten_fields(schema.fields());
    if &fields == schema.fields() {
        return None;
    }
    Some(Arc::new(
        Schema::new(fields).with_metadata(schema.metadata().clone()),
    ))
}

/// Casts each column of `batch` to the corresponding field type in `target`.
/// A no-op fast path is taken when the batch already matches. Used both to
/// flatten (encode side) and to restore (decode side).
pub fn cast_batch_to_schema(batch: RecordBatch, target: &SchemaRef) -> Result<RecordBatch> {
    if batch.schema_ref() == target {
        return Ok(batch);
    }
    let columns = batch
        .columns()
        .iter()
        .zip(target.fields())
        .map(|(column, field)| {
            if column.data_type() == field.data_type() {
                Ok(Arc::clone(column))
            } else {
                Ok(cast(column, field.data_type())?)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(RecordBatch::try_new_with_options(
        Arc::clone(target),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

/// Wraps a decoded record-batch stream so every batch is cast back to `target`,
/// restoring the declared (possibly dict-of-dictionary) schema that was
/// flattened for the IPC wire encoding. A no-op fast path is taken per batch
/// when nothing was flattened.
pub fn restore_record_batch_stream<S>(stream: S, target: SchemaRef) -> SendableRecordBatchStream
where
    S: Stream<Item = Result<RecordBatch, DataFusionError>> + Send + 'static,
{
    let schema = Arc::clone(&target);
    let mapped = stream.map(move |res| res.and_then(|batch| cast_batch_to_schema(batch, &target)));
    Box::pin(RecordBatchStreamAdapter::new(schema, mapped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{ArrayRef, DictionaryArray, StringArray, UInt32Array};
    use datafusion::arrow::datatypes::{Field, UInt32Type};

    fn dict_of_dict() -> ArrayRef {
        let values = Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef;
        let inner = Arc::new(DictionaryArray::<UInt32Type>::new(
            UInt32Array::from(vec![0u32, 1, 2, 0]),
            values,
        )) as ArrayRef;
        Arc::new(DictionaryArray::<UInt32Type>::new(
            UInt32Array::from(vec![0u32, 1, 2, 3]),
            inner,
        )) as ArrayRef
    }

    #[test]
    fn flattens_and_restores_dict_of_dict() {
        let single = DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8));
        let dod = DataType::Dictionary(Box::new(DataType::UInt32), Box::new(single.clone()));
        assert_eq!(flatten_data_type(&dod), single);

        let original = dict_of_dict();
        let declared = Arc::new(Schema::new(vec![Field::new("f", dod, true)]));
        let flat = flatten_schema(&declared).expect("schema has nested dict");

        let batch =
            RecordBatch::try_new(Arc::clone(&declared), vec![Arc::clone(&original)]).unwrap();
        let flattened = cast_batch_to_schema(batch, &flat).unwrap();
        assert_eq!(flattened.column(0).data_type(), &single);

        let restored = cast_batch_to_schema(flattened, &declared).unwrap();
        assert_eq!(restored.schema_ref(), &declared);
        // Values survive the round-trip (compare in the flattened encoding).
        assert_eq!(
            cast(restored.column(0), &single).unwrap().as_ref(),
            cast(&original, &single).unwrap().as_ref(),
        );
    }

    #[test]
    fn no_nested_dict_is_skipped() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new(
                "b",
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        assert!(flatten_schema(&schema).is_none());
    }

    /// Regression test for the dict-of-dictionary IPC bug: a
    /// `Dictionary(K, Dictionary(K2, Utf8))` column cannot survive an Arrow IPC
    /// round-trip as-is (the inner dictionary has no slot in the IPC schema, so
    /// the reader mis-counts buffers), but flattening before encode and
    /// restoring after decode makes it round-trip losslessly.
    #[test]
    fn dict_of_dict_ipc_error_is_fixed_by_flatten_restore() {
        use arrow_ipc::reader::StreamReader;
        use arrow_ipc::writer::StreamWriter;

        fn ipc_roundtrip(
            batch: &RecordBatch,
        ) -> std::result::Result<RecordBatch, datafusion::arrow::error::ArrowError> {
            let mut buf = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buf, &batch.schema())?;
                writer.write(batch)?;
                writer.finish()?;
            }
            StreamReader::try_new(buf.as_slice(), None)?
                .next()
                .expect("one batch")
        }

        let single = DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8));
        let dod = DataType::Dictionary(Box::new(DataType::UInt32), Box::new(single.clone()));
        let original = dict_of_dict();
        let declared = Arc::new(Schema::new(vec![Field::new("f", dod, true)]));
        let batch =
            RecordBatch::try_new(Arc::clone(&declared), vec![Arc::clone(&original)]).unwrap();

        // Reproduces the bug: dict-of-dict cannot round-trip through Arrow IPC.
        let err = ipc_roundtrip(&batch).unwrap_err();
        assert!(
            err.to_string().contains("Buffer count mismatched"),
            "expected the dict-of-dict IPC buffer error, got: {err}"
        );

        // The fix: flatten (encode side) -> IPC -> restore (decode side).
        let flat_schema = flatten_schema(&declared).expect("schema has nested dict");
        let flattened = cast_batch_to_schema(batch, &flat_schema).unwrap();
        let decoded = ipc_roundtrip(&flattened).expect("flattened dict survives IPC");
        let restored = cast_batch_to_schema(decoded, &declared).unwrap();

        assert_eq!(restored.schema_ref(), &declared);
        assert_eq!(
            cast(restored.column(0), &single).unwrap().as_ref(),
            cast(&original, &single).unwrap().as_ref(),
        );
    }
}
