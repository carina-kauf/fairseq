SET=test #TODO CK used to be valid
CHECKPOINT_PATH=discrete_prosody_shift_1_1.pt
DATA=data_config.json

python examples/textless_nlp/pgslm/eval/cont_metrics.py $DATA \
  --metric=teacher_force_everything \
  --path=$CHECKPOINT_PATH \
  --batch-size=16 \
  --fp16 \
  --seed=111 \
  --eval-subset=$SET \
  --f0-discretization-bounds=preprocessing_outputs/mean_norm_log_f0_bin.th --dequantize-prosody