#hacked this by running quantize_with_kmeans.py directly with relevant flags
export PYTHONPYTH=$PYTHONPATH:/Users/carinakauf/repos/fairseq/

MANIFEST=/Users/carinakauf/repos/fairseq/manifests/train/train.tsv #this is the test data we wanna do inference on
OUT_QUANTIZED_FILE=/Users/carinakauf/repos/fairseq/manifests/train/units

#/Users/carinakauf/repos/fairseq/examples/textless_nlp/gslm/speech2unit/clustering/quant.

echo $MANIFEST

cd $PYTHONPYTH

python $(pwd)/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path km.bin \
    --acoustic_model_path hubert_base_ls960.pt \
    --layer 6 \
    --manifest_path /Users/carinakauf/repos/fairseq/manifests/train/train.tsv \
    --out_quantized_file_path /Users/carinakauf/repos/fairseq/manifests/train/units_bash