# ./TrackerV2 <config_file> <image_directory> <groundtruth_file> [save_path]
#   config_file: key=value (see config/lightfc.cfg); model_dir inside it points at the 3 compiled
#   models for YOUR board, template_factor must be set from the experiment yaml.
# Example:
#   ./TrackerV2 config/lightfc.cfg /path/seq/img /path/seq/groundtruth_rect.txt test_results/seq

CONFIG=${1:-config/lightfc.cfg}
IMAGES=${2:?image directory}
GT=${3:?groundtruth file}
OUT=${4:-}

if [ -n "$OUT" ]; then
    mkdir -p "$OUT"
    ./TrackerV2 "$CONFIG" "$IMAGES" "$GT" "$OUT"
else
    ./TrackerV2 "$CONFIG" "$IMAGES" "$GT"
fi
