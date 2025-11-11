#!/bin/bash

# =============================
# 🛠 Configuration
# =============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/../src/tools/simple_fit_bspline.py"   # ← 你的单线程脚本路径

INPUT_DIR=""
OUTPUT_DIR=""
NUM_PROCESSES=4
TIMEOUT_SEC=300  # 5 minutes per JSON file
SAVE_POINTS=false
VERBOSE=false

# =============================
# 📋 Parse arguments
# =============================
usage() {
    echo "Usage: $0 --input DIR --output DIR [options]"
    echo "Options:"
    echo "  --input DIR        Input directory with .json files"
    echo "  --output DIR       Output directory"
    echo "  --workers N        Number of parallel workers (default: 4)"
    echo "  --timeout SEC      Timeout per file in seconds (default: 300)"
    echo "  --save-points      Pass --save_points to Python script"
    echo "  --verbose          Enable verbose logging"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --workers) NUM_PROCESSES="$2"; shift 2 ;;
        --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
        --save-points) SAVE_POINTS=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate
[[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]] && { echo "Error: --input and --output required"; usage; }
[[ ! -d "$INPUT_DIR" ]] && { echo "Error: Input directory not found: $INPUT_DIR"; exit 1; }

# Resolve paths
INPUT_DIR="$(realpath "$INPUT_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"

echo "📁 Input:  $INPUT_DIR"
echo "📁 Output: $OUTPUT_DIR"
echo "⚙️  Workers: $NUM_PROCESSES"
echo "⏱️  Timeout: ${TIMEOUT_SEC}s per file"
[[ "$SAVE_POINTS" == "true" ]] && echo "💾 Save points: enabled"

# =============================
# 📜 Step 1: Find all .json files
# =============================
TMPDIR=$(mktemp -d)
echo "TMPDIR: $TMPDIR"
TASK_LIST="$TMPDIR/tasks.txt"
LOG_DIR="$TMPDIR/logs"
mkdir -p "$LOG_DIR"

echo "🔍 Scanning for .json files..."
find "$INPUT_DIR" -type f -name "*.json" | while read -r json; do
    # Compute relative path
    rel_path="${json#$INPUT_DIR/}"
    out_json="$OUTPUT_DIR/$rel_path"
    out_npz="${out_json%.json}.npz"

    # Skip if both output files exist (idempotent)
    if [[ -f "$out_json" && -f "$out_npz" ]]; then
        [[ "$VERBOSE" == "true" ]] && echo "⏭️  Skipping (already done): $rel_path"
        continue
    fi

    echo "$json|$rel_path" >> "$TASK_LIST"
done

TOTAL_TASKS=$(wc -l < "$TASK_LIST" 2>/dev/null | tr -d ' ')
[[ -z "$TOTAL_TASKS" || "$TOTAL_TASKS" -eq 0 ]] && { echo "✅ No tasks (all files already processed)."; rm -rf "$TMPDIR"; exit 0; }

echo "📋 Found $TOTAL_TASKS tasks."

# =============================
# 🧵 Step 2: Worker function
# =============================
worker() {
    local id=$1
    local task_file=$2
    local log_file=$3

    # Make this subshell exit on SIGTERM (graceful shutdown)
    trap 'echo "[$(date)] Worker $id received SIGTERM, exiting."; exit 143' TERM

    exec > "$log_file" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker $id started."
    echo "    Task file: $task_file"
    echo "    Line count: $(wc -l < "$task_file" 2>/dev/null || echo 0)"
    [[ ! -f "$task_file" ]] && { echo "❌ Task file not found!"; return 1; }
    [[ ! -s "$task_file" ]] && { echo "ℹ️  Empty task file."; return 0; }

    while IFS='|' read -r json_path rel_path; do
        [[ -z "$json_path" ]] && continue

        out_subdir="$OUTPUT_DIR/$(dirname "$rel_path")"
        out_base="$OUTPUT_DIR/$rel_path"
        out_json="${out_base%.json}.json"
        out_npz="${out_base%.json}.npz"

        echo "[$(date '+%H:%M:%S')] Processing: $rel_path"

        # Ensure output dir
        mkdir -p "$out_subdir"

        # 🔔 Run with timeout
        echo python "$PYTHON_SCRIPT" \
            --input-file "$json_path" \
            --output-dir "$out_subdir" \
            ${SAVE_POINTS:+--save-points};

        if timeout "$TIMEOUT_SEC" python "$PYTHON_SCRIPT" \
            --input-file "$json_path" \
            --output-dir "$out_subdir" \
            ${SAVE_POINTS:+--save-points}; then

            # Check if outputs exist
            if [[ -f "$out_json" && -f "$out_npz" ]]; then
                echo "✅ SUCCESS: $rel_path"
                echo "success|$rel_path" >> "$TMPDIR/results.txt"
            else
                echo "❌ FAILED (no output): $rel_path"
                echo "failed|$rel_path" >> "$TMPDIR/results.txt"
            fi
        else
            # timeout or non-zero exit
            if [[ $? -eq 124 ]]; then
                echo "⏱️  TIMEOUT ($TIMEOUT_SEC s): $rel_path"
                echo "timeout|$rel_path" >> "$TMPDIR/results.txt"
            else
                echo "💥 CRASH: $rel_path"
                echo "crash|$rel_path" >> "$TMPDIR/results.txt"
            fi
        fi

    done < "$task_file"

    echo "[$(date '+%H:%M:%S')] Worker $id finished."
}

# =============================
# =============================
# 🚀 Step 3: Split tasks & launch workers (Portable version)
# =============================
# Count lines
TOTAL_TASKS=$(wc -l < "$TASK_LIST" | tr -d ' ')
if [[ -z "$TOTAL_TASKS" || "$TOTAL_TASKS" -eq 0 ]]; then
    echo "✅ No tasks."
    rm -rf "$TMPDIR"
    exit 0
fi

# Portable task splitting using awk (works on Linux/macOS)
awk -v n="$NUM_PROCESSES" -v outbase="$TMPDIR/task_chunk_" '
    {
        chunk = (NR-1) % n   # strict round-robin
        print $0 > (outbase sprintf("%02d", chunk))
    }
' "$TASK_LIST"

# =============================
# 🧹 Cleanup function
# =============================
cleanup() {
    echo ""
    echo "🧹 Received interrupt. Killing workers..."

    # Kill all workers
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    # Wait up to 5 seconds for graceful exit
    sleep 0.5
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "⏳ Worker $pid still running, sending SIGKILL..."
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done

    wait "${pids[@]}" 2>/dev/null

    echo "✅ All workers terminated."
    
    # Optional: cleanup tmp (uncomment if desired)
    # echo "🗑️  Cleaning up temp dir: $TMPDIR"
    # rm -rf "$TMPDIR"

    exit 130  # Standard exit code for Ctrl+C
}

# Set trap BEFORE launching workers
trap cleanup INT TERM

# =============================
# 🚀 Launch workers
# =============================
pids=()
for ((i=0; i<NUM_PROCESSES; i++)); do
    chunk="$TMPDIR/task_chunk_$(printf "%02d" $i)"
    [[ ! -f "$chunk" ]] && continue
    [[ ! -s "$chunk" ]] && continue

    # Launch worker in background, record PID
    worker "$i" "$chunk" "$LOG_DIR/worker_$i.log" &
    pids+=($!)
done

echo "🚀 Launched $NUM_PROCESSES workers. PIDs: ${pids[*]}"

# =============================
# 📈 Real-time progress + ETA (with interrupt safety)
# =============================
START_TIME=$(date +%s)

print_progress() {
    local done=$1 total=$2 elapsed=$3
    (( total == 0 )) && total=1
    local percent=$((done * 100 / total))
    local rate=$(( elapsed > 0 ? done / elapsed : 1 ))
    local eta=$(( rate > 0 ? (total - done) / rate : 0 ))

    local etah=$((eta / 3600))
    local etam=$(((eta % 3600) / 60))
    local etas=$((eta % 60))
    printf -v eta_str "%02d:%02d:%02d" $etah $etam $etas

    printf "\r[%3d%%] %d/%d done | %d tasks/min | ETA: %s" \
        "$percent" "$done" "$total" "$((rate * 60))" "$eta_str"
}

echo ""
echo "📊 Monitoring progress (Ctrl+C to stop gracefully)..."

# Use a flag file to allow safe exit from loop
RUNNING_FLAG="$TMPDIR/running.flag"
touch "$RUNNING_FLAG"

# Progress loop with interrupt safety
while [[ -f "$RUNNING_FLAG" ]] && kill -0 "${pids[@]}" 2>/dev/null; do
    sleep 1
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    DONE=$(wc -l < "$TMPDIR/results.txt" 2>/dev/null | tr -d ' ' || echo 0)
    print_progress "$DONE" "$TOTAL_TASKS" "$ELAPSED"
done

# Remove flag (in case exited normally)
rm -f "$RUNNING_FLAG"

# Final wait
wait "${pids[@]}" 2>/dev/null

# Final progress
DONE=$(wc -l < "$TMPDIR/results.txt" 2>/dev/null | tr -d ' ' || echo 0)
ELAPSED=$(( $(date +%s) - START_TIME ))
print_progress "$DONE" "$TOTAL_TASKS" "$ELAPSED"
echo ""

# Disable trap for normal exit
trap - INT TERM