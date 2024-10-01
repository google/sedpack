// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::io::Read;

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyString;
pub use shard_generated::sedpack::io::flatbuffer::shardfile::{
    root_as_shard, root_as_shard_unchecked, Attribute, Example, Shard,
};

mod example_iteration;
pub mod parallel_map;
// Import the autogenerated code for parsing a shard represented as a FlatBuffer.
// The schema is available in `src/sedpack/io/flatbuffer/shard.fbs`.
#[allow(dead_code, unused_imports)]
mod shard_generated;

/// Python wrappers around `example_iteration`.
mod static_iter {
    use std::collections::HashMap;

    use numpy::IntoPyArray;
    use pyo3::prelude::*;
    use pyo3::{pyclass, pymethods, PyRefMut};

    use super::example_iteration::ExampleIterator;

    /// Implementation details: The goal is to own the ExampleIterator in Rust and only send
    /// examples to Python. This helps with concurrent reading and parsing of shard files.
    /// Moreover Python code cannot compromise integrity of the data structures.
    ///
    /// - We need support for multiple ExampleIterator's at the same time since during training the
    ///   train and validation split are being read in an interleaved manner. To support this each
    ///   RustIter instance keeps a `static_index` determining which `ExampleIterator` it is using
    ///   (dispatch done using a HashMap).
    /// - Since a `HashMap` cannot be instantiated static we use an LazyLock<Mutex<HashMap>>>.
    /// - Using a mutex to avoid the need to use unsafe for a static mutable variable. The overhead
    ///   should be negligible since only a single thread is expected to access this.
    /// - Python does not guarantee that __del__ is called right away (or at all). Thus RustIter
    ///   also implements a context manager which is guaranteed to call __exit__ and drop memory
    ///   owned by the corresponding ExampleIterator.
    static STATIC_ITERATORS: std::sync::LazyLock<
        std::sync::Mutex<HashMap<usize, ExampleIterator>>,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

    #[pyclass]
    pub struct RustIter {
        /// Which ExampleIterator are we interacting with.
        static_index: usize,
        /// Read only value. For iteration we use this object as a context manager which allows us
        /// to free resources in STATIC_ITERATORS on the call of __exit__.
        ///
        /// Alternatives considered:
        /// - __del__ is not yet supported by pyo3 and also not guaranteed to be called by Python.
        #[pyo3(get)]
        can_iterate: bool,
    }

    impl Iterator for RustIter {
        type Item = <ExampleIterator as Iterator>::Item;

        fn next(&mut self) -> Option<Self::Item> {
            // TODO move println to logging.
            if !self.can_iterate {
                println!(
                    "Use the context manager to enable iteration and guaranteed memory \
                     deallocation"
                );
                return None;
            }
            let mut hash_map = STATIC_ITERATORS.lock().unwrap();
            let iter = hash_map
                .get_mut(&self.static_index)
                .expect("The static_index was not found among the STATIC_ITERATORS.");
            iter.next()
        }
    }

    #[pymethods]
    impl RustIter {
        #[new]
        fn new(files: Vec<String>, repeat: bool, threads: usize) -> Self {
            let static_index = rand::random();
            let mut hash_map = STATIC_ITERATORS.lock().unwrap();
            hash_map.insert(static_index, ExampleIterator::new(files, repeat, threads));

            RustIter { static_index, can_iterate: false }
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }
        fn __next__<'py>(mut slf: PyRefMut<'py, Self>) -> Option<PyObject> {
            match slf.next() {
                None => None,
                Some(result) => {
                    // Prepare data back for Python. Wrap in NumPy arrays so that we are passing
                    // only a pointer and not copying data around. Each
                    // attribute is accepted directly by NumPy. There should be no
                    // additional memory copies.
                    let np_result: Vec<Bound<'py, numpy::PyArray<u8, numpy::Ix1>>> = result
                        .into_iter()
                        .map(numpy::ndarray::Array::from_vec)
                        .map(|x| x.into_pyarray_bound(slf.py()))
                        .collect();

                    Some(np_result.to_object(slf.py()))
                }
            }
        }

        fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
            slf.can_iterate = true;
            slf
        }
        fn __exit__(
            mut slf: PyRefMut<'_, Self>, _exc_type: &Bound<'_, PyAny>, _exc_val: &Bound<'_, PyAny>,
            _exc_tb: &Bound<'_, PyAny>,
        ) {
            slf.can_iterate = false;
            // Drop from STATIC_ITERATORS.
            let mut hash_map = STATIC_ITERATORS.lock().unwrap();
            drop(hash_map.remove(&slf.static_index));
        }
    }
}

/// Iterate a given shard file.
///
/// Assumptions:
/// - The shard file is created using the schema defining `shard_generated`.
/// - When the shard file was created attributes were always added in the same order to each
///   example.
/// - This function does not support variable shaped arguments.
/// - Asserts that each example has the same number of shards.
/// - All attributes are saved in little endian. More specifically no endian-ness swap is performed.
/// - Decompression is not supported.
///
/// Returns: A vector of attributes where each attribute is a vector of bytes concatenated along
/// all examples. Meaning that `result[i]` is the `i`-th attribute bytes concatenated across all
/// examples. Then if there are `E` examples and the attribute `i` is (always) represented by `B`
/// bytes then `result[i][e * B: (e + 1) * B]` are the bytes of example `e` (numbered from zero).
/// Thus NumPy can interpret `result[i]` as a c_contiguous array of shape `(E, *attribute.shape)`
/// (no copy is needed for reshaping).
fn iterate_shard(file_path: &str) -> Vec<Vec<u8>> {
    // TODO the byte vectors should be pre-allocated to ensure alignment of larger types. Usually
    // the alignment is at least 8 bytes and moreover NumPy can deal with unaligned arrays (it is a
    // slowdown).
    let mut result = Vec::new();

    let file_bytes = {
        // TODO compressed file support.
        let mut file = match std::fs::File::open(file_path) {
            Ok(f) => f,
            Err(_e) => return result, // No file no shard.
        };
        let mut file_bytes = Vec::new();
        let _ = file.read_to_end(&mut file_bytes);
        file_bytes
    };

    // A shard is a vector of examples (positive number -- invariant kept by Python code).
    // An example is vector of attributes (the same number of attributes in each example of each
    // shard).
    // An attribute is a vector of bytes.
    // For more details see the FlatBuffers schema.
    //
    // If parsing fails at any time it fails.
    let shard = root_as_shard(&file_bytes).unwrap();
    // Number of examples might be different in different shards.
    let examples = shard.examples().unwrap();

    // Should not happen but there is no control over this invariant in Rust.
    if examples.is_empty() {
        return result;
    }

    let num_attributes = examples.get(0).attributes().unwrap().len();
    result.resize(num_attributes, Vec::new());

    // Parse and save examples.
    for example in examples {
        let attributes = example.attributes().unwrap();
        // Always the same number of attributes.
        assert_eq!(attributes.len(), num_attributes);

        for (attribute_id, result_attribute) in result.iter_mut().enumerate() {
            let attribute_bytes = attributes.get(attribute_id).attribute_bytes().unwrap();

            // This is a memory copy. Thus `result` outlives `file_bytes`.
            result_attribute.extend(attribute_bytes);
        }
    }

    result
}

/// Python wrapper function which returns a tuple of `np.ndarray`s containing attribute bytes.
#[pyfunction]
fn iterate_shard_py<'py>(py: Python<'py>, shard_file: &Bound<'_, PyString>) -> PyResult<PyObject> {
    let result: Vec<Vec<u8>> = iterate_shard(&shard_file.extract::<String>().unwrap());

    // Prepare data back for Python. Wrap in NumPy arrays so that we are passing only a pointer and
    // not copying data around. Each attribute is accepted directly by NumPy. There should be no
    // additional memory copies.
    let np_result: Vec<Bound<'py, numpy::PyArray<u8, numpy::Ix1>>> = result
        .into_iter()
        .map(numpy::ndarray::Array::from_vec)
        .map(|x| x.into_pyarray_bound(py))
        .collect();

    Ok(np_result.to_object(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _sedpack_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(iterate_shard_py, m)?)?;
    m.add_class::<static_iter::RustIter>()?;
    Ok(())
}
