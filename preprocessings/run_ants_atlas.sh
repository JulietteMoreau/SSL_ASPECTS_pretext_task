#!/bin/bash
export ANTSPATH=/opt/ANTs/bin/
export PATH=${ANTSPATH}:$PATH
export ITK_GLOBAL_DEFAULTS_NUMBER_OF_THREADS

for d in 'ASPECTS'
do

#input images directory
ct_dir=/path/to/images/
#atlas directory
atlas_dir=/path/to/ATLAS/ASPECTS/

# iterate over the patients
for p in $(ls /path/to/images/)
do


  if ! ls "/path/to/output/images/" | grep -q "^${p::-7}$"
  then

  echo $p
  mkdir /path/to/output/folder/${p::-7}

  output_dir=/path/to/output/folder/${p::-7}
  # CT on DWI MR1
  ./ants_atlas.sh ${ct_dir}/${p} ${atlas_dir}/T1_bet_resize.nii.gz ${atlas_dir}/ASPECTS_L.nii.gz ${atlas_dir}/ASPECTS_R.nii.gz $output_dir $p

  cp ${ct_dir}/${p} ${output_dir}

fi
done
done
