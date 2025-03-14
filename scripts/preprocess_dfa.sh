# bash prepocess_dfa.sh /data/wlsgur4011/DFA /data2/wlsgur4011/GESI/gsplat/data/DFA_processed

#!/bin/bash
set -e        # exit when error
set -o xtrace # print command


#!/bin/bash
set -e        # exit when error
set -o xtrace # print command

# 인자 개수 체크
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 DATA_DIR OUTPUT_DIR"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"

# 각 데이터 이름 (예: dog) 단위로 반복
for datadir in "$DATA_DIR"/*; do
    if [ -d "$datadir" ]; then
        data_name=$(basename "$datadir")
        # 원본의 Intrinsic.inf와 CamPose.inf 파일 경로
        intrinsic_src="$datadir/Intrinsic.inf"
        campose_src="$datadir/CamPose.inf"
        
        # $datadir/img/ 내의 모든 서브폴더에 대해 반복
        for subfolder in "$datadir/img/"*; do
            if [ -d "$subfolder" ]; then
                echo "Processing subfolder: $(basename "$subfolder") in $data_name"
                # 각 시퀀스 (예: 0, 1, …)에 대해 반복
                for seqdir in "$subfolder"/*; do
                    if [ -d "$seqdir" ]; then
                        seq_name=$(basename "$seqdir")
                        # seq_name이 숫자가 아니거나 10의 배수가 아니라면 건너뜀
                        if ! [[ $seq_name =~ ^[0-9]+$ ]] || [ $((seq_name % 10)) -ne 0 ]; then
                            continue
                        fi
                        echo "Processing frame: $seq_name in subfolder $(basename "$subfolder") for $data_name"
                        # 새 폴더 생성: OUTPUT_DIR/[data_name]/[seq]/images
                        dest_img_dir="$OUTPUT_DIR/$data_name/$seq_name/images"
                        mkdir -p "$dest_img_dir"
                        dest_seq_dir="$OUTPUT_DIR/$data_name/$seq_name"
                        mkdir -p "$dest_seq_dir"

                        # 원본 이미지 폴더 안에서, _alpha가 없는 파일(즉, rgb 이미지) 처리
                        for img_file in "$seqdir"/img_*.png; do
                            if [[ "$img_file" == *_alpha.png ]]; then
                                continue
                            fi
                            base=$(basename "$img_file")
                            base_no_ext="${base%.png}"
                            alpha_file="$seqdir/${base_no_ext}_alpha.png"

                            if [ -f "$alpha_file" ]; then
                                out_file="$dest_img_dir/${base_no_ext}_rgba.png"
                                convert "$img_file" "$alpha_file" -alpha off -compose CopyOpacity -composite "$out_file"
                            else
                                echo "Warning: $alpha_file not found for $img_file"
                            fi
                        done

                        # Intrinsic.inf 복사
                        if [ -f "$intrinsic_src" ]; then
                            cp "$intrinsic_src" "$dest_seq_dir/"
                        else
                            echo "Warning: $intrinsic_src not found."
                        fi

                        # CamPose.inf 복사 후, 이름을 Campose.inf로 변경
                        if [ -f "$campose_src" ]; then
                            cp "$campose_src" "$dest_seq_dir/Campose.inf"
                        else
                            echo "Warning: $campose_src not found."
                        fi
                    fi
                done
            fi
        done
    fi
done

