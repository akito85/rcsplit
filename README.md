Below are CLI usage examples for splitting files using the `cli-rcsplit` tool, tailored to your request. I’ll provide examples for streaming small files (1MB to 10GB) using memory-mapped mode (mmap), splitting very large files (>40GB) using streaming mode, and the most optimized usage for files ranging from 1MB to 40GB.

---

### 1. Splitting Small Files (1MB to 10GB) Using Memory-Mapped Mode (mmap)
For files between 1MB and 10GB, memory-mapped mode (mmap) is ideal because these files can fit comfortably in memory, allowing faster processing without the need for streaming.

- **Example for a 1MB File**:
  ```
  ./target/release/cli-rcsplit -i small_file.txt -p "^>" -o splits -t 2 -b 512 -m 1024
  ```
  - `-i small_file.txt`: Input file (1MB).
  - `-p "^>"`: Split at lines starting with `>` (regex pattern).
  - `-o splits`: Output directory for split files.
  - `-t 2`: Trim 2 characters from each line.
  - `-b 512`: Buffer size of 512 KB for writing.
  - `-m 1024`: Maximum memory of 1024 MB.
  - **Note**: No streaming flag (`-s`) is used, as mmap is faster for small files.

- **Example for a 10GB File**:
  ```
  ./target/release/cli-rcsplit -i medium_file.txt -p "^>" -o splits -t 2 -b 1024 -m 2048
  ```
  - `-i medium_file.txt`: Input file (10GB).
  - `-p "^>"`: Same regex pattern.
  - `-o splits`: Output directory.
  - `-t 2`: Trim 2 characters.
  - `-b 1024`: Larger buffer size (1024 KB) for better performance.
  - `-m 2048`: Increased max memory (2048 MB) to handle the larger file efficiently.
  - **Note**: Still uses mmap (no `-s`) as 10GB is manageable in memory with sufficient resources.

---

### 2. Splitting Large Files (>40GB) Using Streaming Mode
For files exceeding 40GB, streaming mode is necessary to process the file in smaller segments, preventing excessive memory usage.

- **Example for a 40GB+ File**:
  ```
  ./target/release/cli-rcsplit -i large_file.txt -p "^>" -o splits -t 2 -b 512 -m 1024 -s --segment_size 512
  ```
  - `-i large_file.txt`: Input file (e.g., 40GB or more).
  - `-p "^>"`: Regex pattern for splitting.
  - `-o splits`: Output directory.
  - `-t 2`: Trim 2 characters.
  - `-b 512`: Buffer size of 512 KB.
  - `-m 1024`: Max memory of 1024 MB.
  - `-s`: Enables streaming mode for large files.
  - `--segment_size 512`: Processes the file in 512 MB segments, balancing memory usage and performance.

---

### 3. Optimized Usage for Files Between 1MB and 40GB
For the range of 1MB to 40GB, the optimal approach depends on the file size:
- **Files <10GB**: Use memory-mapped mode (mmap) for speed.
- **Files >10GB up to 40GB**: Use streaming mode for efficiency with larger files.

- **Optimized Example for a 5GB File (mmap)**:
  ```
  ./target/release/cli-rcsplit -i file_5gb.txt -p "^>" -o splits -t 2 -b 1024 -m 2048
  ```
  - `-i file_5gb.txt`: Input file (5GB).
  - `-p "^>"`: Regex pattern.
  - `-o splits`: Output directory.
  - `-t 2`: Trim 2 characters.
  - `-b 1024`: Higher buffer size for faster I/O.
  - `-m 2048`: Increased memory for efficient processing.
  - **Why Optimized**: Larger buffer and memory settings leverage mmap’s speed for files under 10GB.

- **Optimized Example for a 20GB File (Streaming)**:
  ```
  ./target/release/cli-rcsplit -i file_20gb.txt -p "^>" -o splits -t 2 -b 512 -m 1024 -s --segment_size 512
  ```
  - `-i file_20gb.txt`: Input file (20GB).
  - `-p "^>"`: Regex pattern.
  - `-o splits`: Output directory.
  - `-t 2`: Trim 2 characters.
  - `-b 512`: Moderate buffer size for streaming.
  - `-m 1024`: Max memory of 1024 MB.
  - `-s`: Streaming mode enabled.
  - `--segment_size 512`: Processes in 512 MB chunks.
  - **Why Optimized**: Streaming with a reasonable segment size ensures efficient memory use for files between 10GB and 40GB.

---

### Key Notes on Optimization
- **Memory-Mapped Mode (mmap)**:
  - Best for files up to 10GB.
  - Use larger buffer sizes (`-b 1024`) and max memory (`-m 2048`) for faster performance.
- **Streaming Mode**:
  - Essential for files over 10GB, especially beyond 40GB.
  - Adjust `--segment_size` (e.g., 256-1024 MB) based on available memory—larger segments improve speed but use more memory.
- **General Tips**:
  - Increase `-b` (buffer size) to reduce I/O operations.
  - Set `-m` (max memory) to a value that fits your system’s capacity for better performance.

These examples provide efficient and practical ways to split files across the specified size ranges! Let me know if you need further assistance.
