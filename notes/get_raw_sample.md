# Random Document Sampling from Large Text Files

## Problem Statement
Need to create a function that can randomly sample N documents from large text datasets where:
- Documents are separated by `<|endoftext|>` tokens
- Files can be very large (11GB+ for owt_train.txt)
- Memory constraints prevent loading entire file into memory
- Function should be fast for repeated sampling

## Dataset Analysis
- TinyStoriesV2-GPT4-valid.txt: ~158k lines, ~27k documents
- owt_train.txt: ~11GB file size
- Documents are variable length stories/texts

## Solution: Two-Pass Strategy

### Pass 1: Build Document Index
- Read through file line-by-line to find `<|endoftext|>` markers
- Record byte positions using `file.tell()` for each document start
- Store positions in simple list: `[pos1, pos2, pos3, ...]`
- Memory footprint: O(num_documents) integers vs O(file_size) text

### Pass 2: Random Sampling
- Use `random.sample()` to select N random indices from position list
- Use `file.seek(position)` to jump directly to selected documents
- Read from position until next `<|endoftext|>` or EOF

## Key Design Decisions

### Index Storage: Byte Positions vs Line Numbers
**Chosen: Byte positions**
- Pros: Direct seeking with `file.seek()`, works across any file structure
- Cons: Slightly more complex to build index
- Alternative: Line numbers require sequential reading to reach target

### Reading Strategy: Line-by-line vs Chunk Reading
**Chosen: Line-by-line for index building**
- Pros: Easy to detect `<|endoftext|>`, clear boundaries
- Cons: Potentially slower than chunk reading
- Trade-off: Simplicity vs performance

### Index Caching
**Optional feature to implement:**
- Cache document positions to disk for large files
- Include file modification time to detect stale cache
- Significant speedup for repeated operations on same file

## Implementation Considerations

### Function Signature
```python
def random_sample_documents(
    file_path: str,
    num_samples: int,
    delimiter: str = "<|endoftext|>",
    use_cache: bool = True
) -> List[str]
```

### Error Handling
- Handle case where `num_samples > total_documents`
- Detect corrupted files (documents too long, missing delimiters)
- Validate file exists and is readable

### Edge Cases
- Last document (no delimiter after it)
- Empty documents
- Files not ending with delimiter
- Very large individual documents

## Technical Details

### File I/O Operations Needed
- `file.tell()`: Get current byte position
- `file.seek(position)`: Jump to specific byte position
- Line-by-line reading for index building
- Efficient document extraction after seeking

### Memory Efficiency
- Index size: ~4 bytes per document (integer positions)
- For 27k documents: ~108KB memory for index
- No need to load document content until sampling

## Alternative Approaches Considered

### Load All Into Memory
- Rejected: Memory constraints with 11GB files
- Would work for smaller datasets but not scalable

### Reservoir Sampling (Single Pass)
- Considered but rejected for this use case
- Would require streaming through entire file each time
- Two-pass approach better for repeated sampling

### Pure Chunk-Based Reading
- More complex boundary handling for `<|endoftext|>` spanning chunks
- Line-by-line approach simpler and sufficient for this use case

## Next Steps
1. Implement core function with two-pass strategy
2. Add comprehensive error handling
3. Implement optional index caching
4. Test with both small (TinyStories) and conceptually large datasets
5. Benchmark performance and memory usage