d_whl=$(ls -t dist/*.whl | head -n 1)

echo ${d_whl}
whl="${1:-$d_whl}"

pip install ${whl} --no-deps #--force-reinstall 

