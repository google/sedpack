// Copyright 2025 Google LLC
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

use std::fs;

use criterion::{criterion_group, criterion_main, Criterion};
use sedpack_rs::example_iteration::{
    get_shard_progress, CompressionType, ExampleIterator, ShardInfo,
};
pub use sedpack_rs::parallel_map::parallel_map;

pub fn get_shard_files() -> Vec<ShardInfo> {
    let dir = "mnist_fb_gzip/train/";
    let shard_infos: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|p| p.ok())
        .map(|p| p.path().to_str().unwrap().to_string())
        .filter(|p| p.ends_with(".fb"))
        .map(|file_path| ShardInfo { file_path, compression_type: CompressionType::Gzip })
        .collect();
    println!(">> Decoding {} shards", shard_infos.len());
    shard_infos
}

pub fn example_iterator_benchmark(c: &mut Criterion) {
    let shard_infos = get_shard_files();
    c.bench_function("ExampleIterator", |b| {
        b.iter(|| {
            for example in ExampleIterator::new(shard_infos.clone(), 12) {
                let _ = std::hint::black_box(example);
            }
        })
    });
}

pub fn parallel_map_benchmark(c: &mut Criterion) {
    let shard_infos = get_shard_files();
    c.bench_function("parallel_map", |b| {
        b.iter(|| {
            for shard in
                parallel_map(|x| get_shard_progress(&x), shard_infos.clone().into_iter(), 32)
            {
                let _ = std::hint::black_box(shard);
            }
        })
    });
}

criterion_group!(benches, parallel_map_benchmark, example_iterator_benchmark);
criterion_main!(benches);
