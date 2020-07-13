source /etc/profile.d/modules.sh
module load singularity/2.6.1
cd /home/acb11319bq/ito/proj_aopa/proj_aopa/ito
singularity exec -B /groups2/gcb50278 --nv ./medical.img python3 deeplab_predict.py
