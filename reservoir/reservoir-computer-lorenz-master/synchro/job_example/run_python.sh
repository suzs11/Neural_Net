#!/home/lenovo/Software/anaconda3/bin/python
#SBATCH -J pythontest                                           
#SBATCH --ntasks=8                                 
#SBATCH --nodes=1                                    
#SBATCH --ntasks-per-node=20                      
#SBATHC --time=00:30:00
#SBATCH -p low                                                
#SBATCH --output=%j.log                                     
#SBATCH --job-name=PythonTest
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=suzs11@163.com


# load the environment
#module purge
#module load /public/home/qushixian_st3/anaconda3/bin/python3.8

# run python
python kuramoto_osic.py
