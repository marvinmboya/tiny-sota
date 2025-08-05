msg='tiny sota built!'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

SUCCESS() {
    echo "${GREEN} $1 ${NC}"
}

ERROR() {
    echo "${RED} $1 ${NC}"
}

SUCCESS "building tiny-sota..."
python -m build
if [ $? -eq 0 ]; then
    SUCCESS "build tool done..."
    SUCCESS "$msg"
else
    ERROR "build tool err..."
    SUCCESS "installing build tool"
    pip install --upgrade build 
    python -m build
    echo "$msg"
fi
