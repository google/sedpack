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

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sedpack_rs::example_iteration::{CompressionType, ExampleIterator, ShardInfo, get_shard_progress};
pub use sedpack_rs::parallel_map::parallel_map;
use std::fs;

pub fn get_shard_files() -> Vec<ShardInfo> {
    let dir = "mnist_fb_gzip/train/";
    let shard_infos: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter(|p| p.is_ok())
        .map(|p| p.expect("filtered"))
        .map(|p| p.path().to_str().unwrap().to_string())
        .filter(|p| p.ends_with(".fb"))
        .into_iter()
        .map(|file_path| ShardInfo { file_path: file_path, compression_type: CompressionType::Gzip})
        .collect();
    println!(">> Decoding {} shards", shard_infos.len());
    shard_infos
}

pub fn example_iterator_benchmark(c: &mut Criterion) {
    let shard_infos = get_shard_files();
    c.bench_function("ExampleIterator", |b| b.iter(|| {
        for _example in ExampleIterator::new(black_box(shard_infos.clone()), false, 12) {}
    }));
}

pub fn parallel_map_benchmark(c: &mut Criterion) {
    let shard_infos = get_shard_files();
    c.bench_function("parallel_map", |b| b.iter(|| {
        for _shard in parallel_map(|x| get_shard_progress(&x), black_box(shard_infos.clone().into_iter()), 32) {}
    }));
}

criterion_group!(benches, parallel_map_benchmark, example_iterator_benchmark);
criterion_main!(benches);
