#!/bin/bash
export ANTSPATH=/opt/ANTs/bin/
export PATH=${ANTSPATH}:$PATH
export ITK_GLOBAL_DEFAULTS_NUMBER_OF_THREADS


start=$SECONDS

f=$1 ; m=$2; mm1=$3; mm2=$4
output_dir=${5}
p=${6}
pat=` basename $f | cut -d '.' -f 1 `


if [[ ! -s $f ]] ; then echo no fixed $f ; exit; fi
if [[ ! -s $m ]] ; then echo no moving $m ;exit; fi


##########################
# LINEAR_INTERPOLATION
##########################
var_f=$(fslinfo $1 | sed -n 7p | awk '{print $2}')
ResampleImage 3 $2 atlas_resample.nii.gz $var_f'x'$var_f'x'$var_f
ResampleImage 3 ${mm1} ASPECTS_L.nii.gz $var_f'x'$var_f'x'$var_f
ResampleImage 3 ${mm2} ASPECTS_R.nii.gz $var_f'x'$var_f'x'$var_f

##########################
# REGISTRATION
##########################
dim=3 # image dimensionality
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=40  # controls multi-threading

reg=${AP}antsRegistration           # path to antsRegistration

its=400x250x10
percentage=0.3
syn="10000x10000x10000x10000x5000x3000x1000x500x100,1e-6,5"


f=$1
m=atlas_resample.nii.gz
nm=${pat}_fixed_atlas_moving   # construct output prefix
echo affine $m $f outname is ${nm}


$reg -d $dim -r [ $f, $m ,1]  \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t translation[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 8x4x2 -l 1 \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t rigid[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 8x4x2 -l 1 \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t affine[ 0.1 ] \
                         -c [$its,1.e-8,20]  \
                        -s 4x2x1vox  \
                        -f 4x2x1 -l 1 \
                        -m mattes[  $f, $m , 1 , 32, regular, $percentage ] \
                         -t SyN[ 0.1, 3, 0 ] \
                         -c [ $syn ]  \
                        -s 2x2x2x1vox  \
                        -f 4x2x1 -l 1 -u 1 -z 1 \
                        -o [${nm}]



f=$1
m=atlas_resample.nii.gz
mm1=ASPECTS_L.nii.gz; mm2=ASPECTS_R.nii.gz


antsApplyTransforms -d $dim -i $m -r $f -n linear -t ${nm}1Warp.nii.gz -t ${nm}0GenericAffine.mat -o ${output_dir}/atlas_${p}
antsApplyTransforms -d $dim -i ${mm1} -r $f -n linear -t ${nm}1Warp.nii.gz -t ${nm}0GenericAffine.mat -o ${output_dir}/${mm1} --interpolation NearestNeighbor
antsApplyTransforms -d $dim -i ${mm2} -r $f -n linear -t ${nm}1Warp.nii.gz -t ${nm}0GenericAffine.mat -o ${output_dir}/${mm2} --interpolation NearestNeighbor

rm atlas_resample.nii.gz
rm ${nm}1Warp.nii.gz ${nm}1InverseWarp.nii.gz
mv ${nm}0GenericAffine.mat ${output_dir}/0GenericAffine.mat


end=$SECONDS
duration=$(( end - start ))
echo job took $duration seconds
