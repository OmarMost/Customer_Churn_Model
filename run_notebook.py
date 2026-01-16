import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

notebook_filename = 'churn_analysis.ipynb'
path = 'd:/Model/Customer_Churn_Model/'

try:
    with open(path + notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': path}})
    
    with open(path + 'churn_analysis_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print('Notebook executed successfully.')
except Exception as e:
    print(f"Error executing notebook: {e}")
