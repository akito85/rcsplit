use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::{Context, Result};
use clap::Parser;
use memmap2::Mmap;
use rayon::prelude::*;
use regex_automata::{
    dfa::{dense, Automaton},
    Anchored, Input,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    pattern: String,
    #[arg(short, long, default_value = "out_chunks")]
    output: String,
    #[arg(short, long, default_value_t = 4)]
    trim: usize,
    #[arg(short, long, default_value_t = 256)]
    buffer_size: usize,
    #[arg(short, long, default_value_t = 512)]
    max_memory: usize,
    #[arg(short, long)]
    streaming: bool,
    #[arg(short, long, default_value_t = 256)]
    segment_size: usize,
}

struct Chunk {
    start: usize,
    end: usize,
    chunk_num: usize,
}

fn process_chunk(
    data: &[u8],
    chunk: Chunk,
    output_dir: &PathBuf,
    trim: usize,
    buffer_size: usize,
) -> Result<()> {
    let chunk_data = &data[chunk.start..chunk.end];
    let out_path = output_dir.join(format!("chunk_{:05}.txt", chunk.chunk_num));
    let file = File::create(&out_path)?;
    let mut writer = BufWriter::with_capacity(buffer_size * 1024, file);

    let mut lines = chunk_data.split(|&c| c == b'\n');
    lines.next();

    for line in lines {
        if line.len() > trim {
            writer.write_all(&line[trim..])?;
            writer.write_all(b"\n")?;
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_path = PathBuf::from(&args.input);
    let output_dir = PathBuf::from(&args.output);
    fs::create_dir_all(&output_dir)?;

    let file = File::open(&input_path)?;
    let file_size = file.metadata()?.len() as usize;

    // Build optimized DFA with byte-oriented configuration
    let dfa = Arc::new(
        dense::Builder::new()
            .configure(
                dense::Config::new()
                    .byte_classes(false)
                    .start_kind(regex_automata::dfa::StartKind::Anchored)
            )
            .build(&args.pattern)
            .context("Failed to build DFA")?,
    );

    if args.streaming || file_size > 10 * 1024 * 1024 * 1024 {
        process_streaming(file, file_size, &output_dir, args, &dfa)
    } else {
        process_mapped(file, file_size, &output_dir, args, &dfa)
    }
}

fn process_mapped<A: Automaton + Send + Sync>(
    file: File,
    file_size: usize,
    output_dir: &PathBuf,
    args: Args,
    dfa: &Arc<A>,
) -> Result<()> {
    let mmap = unsafe { Mmap::map(&file)? };
    let counter = Arc::new(AtomicUsize::new(0));

    let positions: Vec<usize> = (0..file_size)
        .into_par_iter()
        .fold(
            || Vec::with_capacity(1024),
            |mut acc, pos| {
                if pos == 0 || mmap[pos - 1] == b'\n' {
                    let input = Input::new(&mmap[pos..])
                        .anchored(Anchored::Yes)
                        .range(..file_size - pos);
                    match dfa.try_search_fwd(&input) {
                        Ok(Some(_)) => {
                            acc.push(pos);
                        }
                        _ => (),
                    }
                }
                acc
            },
        )
        .flatten()
        .collect();

    let mut positions = positions;
    positions.sort_unstable();
    positions.dedup();
    positions.push(file_size);

    positions.par_windows(2)
        .enumerate()
        .try_for_each(|(_, window)| {
            let chunk_num = counter.fetch_add(1, Ordering::Relaxed);
            let chunk = Chunk {
                start: window[0],
                end: window[1],
                chunk_num,
            };
            process_chunk(&mmap, chunk, output_dir, args.trim, args.buffer_size)
        })?;

    println!("Processed {} chunks", positions.len() - 1);
    Ok(())
}

fn process_streaming<A: Automaton + Send + Sync>(
    file: File,
    _file_size: usize,
    output_dir: &PathBuf,
    args: Args,
    dfa: &Arc<A>,
) -> Result<()> {
    let segment_size = args.segment_size * 1024 * 1024;
    let mut reader = BufReader::with_capacity(segment_size, file);
    let counter = Arc::new(AtomicUsize::new(0));

    rayon::scope(|s| {
        let mut buffer = Vec::with_capacity(segment_size);
        let mut leftovers = Vec::new();

        loop {
            let read_size = segment_size - buffer.len();
            let mut chunk = vec![0; read_size];
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                Err(e) => {
                    eprintln!("Read error: {}", e);
                    break;
                }
            }

            let mut positions = vec![0];
            let mut cursor = 0;
            while cursor < buffer.len() {
                let input = Input::new(&buffer[cursor..])
                    .anchored(Anchored::Yes);
                match dfa.try_search_fwd(&input) {
                    Ok(Some(_)) => {
                        positions.push(cursor);
                    }
                    _ => {},
                }
                cursor += 1;
            }

            let chunks: Vec<_> = positions.windows(2)
                .map(|w| (w[0], w[1]))
                .collect();

            let output_dir = output_dir.clone();
            let buffer_arc: Arc<[u8]> = Arc::from(buffer.as_slice());
            let counter_clone = counter.clone();

            s.spawn(move |_| {
                chunks.into_par_iter()
                    .for_each(|(start, end)| {
                        let chunk_num = counter_clone.fetch_add(1, Ordering::Relaxed);
                        let chunk = Chunk {
                            start,
                            end,
                            chunk_num,
                        };

                        if let Err(e) = process_chunk(
                            &*buffer_arc,
                            chunk,
                            &output_dir,
                            args.trim,
                            args.buffer_size,
                        ) {
                            eprintln!("Chunk error: {}", e);
                        }
                    });
            });

            leftovers = if let Some(last) = positions.last() {
                buffer[*last..].to_vec()
            } else {
                buffer.clone()
            };
            buffer = leftovers.clone();
        }

        // Process final leftover data
        if !buffer.is_empty() {
            let output_dir = output_dir.clone();
            let buffer_arc: Arc<[u8]> = Arc::from(buffer.as_slice());
            let counter_clone = counter.clone();
            s.spawn(move |_| {
                let chunk_num = counter_clone.fetch_add(1, Ordering::Relaxed);
                let chunk = Chunk {
                    start: 0,
                    end: buffer_arc.len(),
                    chunk_num,
                };
                process_chunk(&*buffer_arc, chunk, &output_dir, args.trim, args.buffer_size)
                    .unwrap_or_else(|e| eprintln!("Final chunk error: {}", e));
            });
        }
    });

    Ok(())
}
