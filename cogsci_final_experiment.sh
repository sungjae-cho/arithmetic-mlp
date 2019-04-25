exec &> "log_cogsci2019.txt"

sleep_sec=10
experiment_name='cogsci2019_final'
operand_digits=4
hidden_units=64

start_echo()
{
  i=$1
  operator=$2
  echo "==================================================================="
  echo "Run the $i-th training of ${operand_digits} digit ${operator}"
}

experiment()
{
  device_num=$1
  operator='add'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='add'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='subtract'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='subtract'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='divide'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='divide'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='modulo'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='modulo'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='multiply'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
  operator='multiply'
  python3 mlp_run.py $experiment_name $operand_digits $operator $hidden_units $device_num
}

for j in {1..60..1}
  do
    device_num=$(( $j % 5 ))
    experiment $device_num &
    sleep $sleep_sec
  done
