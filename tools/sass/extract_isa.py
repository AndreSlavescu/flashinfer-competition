"""
Parse intercepted nvdisasm output to extract SASS instruction documentation.

Parsing logic adapted from: https://github.com/0xD0GF00D/DocumentSASS

nvdisasm embeds two key string tables:
  - 'ARCHITECTURE' — instruction descriptions for the target compute capability
  - 'OPERATION SETS' — scheduling latency data per instruction

This script reads the raw output from running nvdisasm under LD_PRELOAD with
the memcpy interceptor, locates these string tables by matching delimiters,
and extracts them into structured text files.

Usage:
    python3 extract_isa.py raw_intercept_output.txt
    # Produces: raw_intercept_output_instructions.txt
    #           raw_intercept_output_latencies.txt
"""

import re
import sys

# Delimiter format emitted by intercept.c: <dest_ptr src_ptr byte_count>
DELIMITER = re.compile(r"<0x[\da-f]+ 0x[\da-f]+ [\d]+>")


def _get_key(entry: str):
    """Unpack '<0xDEST 0xSRC SIZE>' into (dest, src, size) strings."""
    return entry[1:-1].split()


def _get_string(data: str, src: str, first_line: bool = False) -> str:
    """
    Collect all text fragments associated with a particular source pointer.

    The interceptor emits delimiter lines followed by the raw bytes. This
    function collects the non-delimiter lines that follow delimiters whose
    source pointer matches `src`.
    """
    collect = False
    partial = []
    full = []

    for line in data.splitlines() + ["<0x0 0x0 0>"]:
        if DELIMITER.match(line):
            collect = line.split()[1] == src
            full.append("\n".join(partial))
            partial.clear()
        elif collect:
            if first_line:
                return line
            partial.append(line)

    return "".join(full)


def _get_file(data: str, name: str) -> str:
    """Find a named section in the intercepted output and extract its full content."""
    match = re.search(DELIMITER.pattern + r"\n" + re.escape(name), data)
    if match is None:
        return ""
    _, src, _ = _get_key(match.group().splitlines()[0])
    return _get_string(data, src)


def extract_from_raw(raw_text: str) -> dict:
    """
    Extract instruction docs and latencies from raw intercepted nvdisasm output.

    Returns:
        {"instructions": str, "latencies": str}
    """
    instructions = _get_file(raw_text, "ARCHITECTURE")
    if instructions.endswith("\n\n"):
        instructions = instructions[:-2]

    latencies = _get_file(raw_text, "OPERATION SETS")

    return {"instructions": instructions, "latencies": latencies}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_isa.py <raw_intercept_output.txt> [...]")
        sys.exit(1)

    for fname in sys.argv[1:]:
        with open(fname, "r", encoding="ascii", errors="replace") as f:
            data = f.read()

        base = fname
        for suffix in (".txt", "_intercept"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]

        result = extract_from_raw(data)

        if result["instructions"]:
            out_path = f"{base}_instructions.txt"
            with open(out_path, "w") as f:
                f.write(result["instructions"])
            print(f"  {out_path}: {len(result['instructions'])} bytes")

        if result["latencies"]:
            out_path = f"{base}_latencies.txt"
            with open(out_path, "w") as f:
                f.write(result["latencies"])
            print(f"  {out_path}: {len(result['latencies'])} bytes")

        if not result["instructions"] and not result["latencies"]:
            print(f"  {fname}: no ISA metadata found (nvdisasm format may have changed)")
