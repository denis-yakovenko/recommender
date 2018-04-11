"""
Code that goes along with the Airflow located at:
http://airflow.readthedocs.org/en/latest/tutorial.html
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2018, 4, 11),
    #'email': ['airflow@example.com'],
    #'email_on_failure': False,
    #'email_on_retry': False,
    #'retries': 1,
    #'retry_delay': timedelta(minutes=5),
    #'dag.catchup': False,
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG('recommender', default_args=default_args, catchup=False, schedule_interval='@daily')

# t1, t2 and t3 are examples of tasks created by instantiating operators

t1 = BashOperator(
    task_id='PrepareDataSet',
    bash_command='/home/den/Recommender/PrepareDataSet.sh ',
    dag=dag)

t2 = BashOperator(
    task_id='DesignModel',
    bash_command='/home/den/Recommender/DesignModel.sh ',
    dag=dag)

t3 = BashOperator(
    task_id='TrainModel',
    bash_command='/home/den/Recommender/TrainModel.sh ',
    dag=dag)

t4 = BashOperator(
    task_id='Predict',
    bash_command='/home/den/Recommender/Predict.sh ',
    dag=dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
t4.set_upstream(t3)
