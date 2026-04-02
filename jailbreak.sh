# Reading dataset folder from params.py
dataset_folder=$(python3 -c "import utils.params as params; print(params.dataset_folder)")

### Aboships

aboships_folder="$dataset_folder/aboships/images"

for txt_file in "$aboships_folder"/*_jb.txt; do
    [ -f "$txt_file" ] || continue
    goal=$(grep -E "^GOAL:" "$txt_file" | sed 's/^GOAL:[[:space:]]*//')
    target=$(grep -E "^TARGET:" "$txt_file" | sed 's/^TARGET:[[:space:]]*//')
    mp4_file="${txt_file%_jb.txt}.mp4"
    if [ -n "$goal" ] && [ -n "$target" ]; then
        python robopair/main.py \
            --goal "$goal" \
            --target-str "$target" \
            --video-path "$mp4_file" \
            --robot "dolphin" \
            --target-model "dolphin"
    else
        echo "[!] Missing GOAL or TARGET in $txt_file"
    fi
done

### LARS

lars_dir_test="$dataset_folder/lars/test/images_seq"
lars_dir_train="$dataset_folder/lars/train/images_seq"
lars_dir_val="$dataset_folder/lars/val/images_seq"

lars_dirs=("$lars_dir_test" "$lars_dir_train" "$lars_dir_val")

for lars_dir in "${lars_dirs[@]}"; do
    for txt_file in "$lars_dir"/*_jb.txt; do
        [ -f "$txt_file" ] || continue
        goal=$(grep -E "^GOAL:" "$txt_file" | sed 's/^GOAL:[[:space:]]*//')
        target=$(grep -E "^TARGET:" "$txt_file" | sed 's/^TARGET:[[:space:]]*//')
        mp4_file="${txt_file%_jb.txt}.mp4"
        if [ -n "$goal" ] && [ -n "$target" ]; then
            python robopair/main.py \
                --goal "$goal" \
                --target-str "$target" \
                --video-path "$mp4_file" \
                --robot "dolphin" \
                --target-model "dolphin"
        else
            echo "[!] Missing GOAL or TARGET in $txt_file"
        fi
    done
done

### NuScenes

nuscenes_dir="$dataset_folder/nuscenes/videos"

for folder in "$nuscenes_dir"/*; do
    if [ -d "$folder" ]; then
        goal_target_file="$folder/jb.txt"

        if [ -f "$goal_target_file" ]; then
            goal=$(grep -E "^GOAL:" "$goal_target_file" | sed 's/^GOAL:[[:space:]]*//')
            target=$(grep -E "^TARGET:" "$goal_target_file" | sed 's/^TARGET:[[:space:]]*//')

            if [ -n "$goal" ] && [ -n "$target" ]; then
                echo "[+] Running for: $(basename "$folder")"
                python robopair/main.py \
                    --goal "$goal" \
                    --target-str "$target" \
                    --video-path "$folder/video.mp4" \
                    --robot "dolphin" \
                    --target-model "dolphin"
            else
                echo "[!] Missing GOAL or TARGET in $goal_target_file"
            fi
        else
            echo "[!] No goal_target.txt in $folder"
        fi
    fi
done
