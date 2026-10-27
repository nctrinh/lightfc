#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1
CXX=${CXX:-g++}
os=`lsb_release -a | grep "Distributor ID" | sed 's/^.*:\s*//'`
os_version=`lsb_release -a | grep "Release" | sed 's/^.*:\s*//'`
arch=`uname -p`
target_info=${os}.${os_version}.${arch}
install_prefix_default=$HOME/.local/${target_info}
$CXX --version

result=0 && pkg-config --list-all | grep opencv4 && result=1
if [ $result -eq 1 ]; then
     OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv4)
else
     OPENCV_FLAGS=$(pkg-config --cflags --libs-only-L opencv)
fi

# ./build.sh test : host-only tests (OpenCV only, no Vitis AI), see test/ and tools/make_golden.py
if [ "$1" == "test" ]; then
     TEST_FLAGS="-O2 -std=c++17 -Wall -I$PWD/src/tracker -I$PWD/src/DPU $(pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)"
     TEST_LIBS="$(pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv) -lpthread"
     $CXX $TEST_FLAGS -o test_host_ops $PWD/test/test_host_ops.cpp $PWD/src/tracker/lightfc_ops.cpp $TEST_LIBS || exit 1
     $CXX $TEST_FLAGS -o test_tracker_flow $PWD/test/test_tracker_flow.cpp \
          $PWD/src/tracker/AItracker.cpp $PWD/src/tracker/tracker.cpp $PWD/src/tracker/lightfc_ops.cpp \
          $PWD/src/tracker/lightfc_config.cpp $PWD/src/DPU/model_runner.cpp $TEST_LIBS || exit 1
     echo "built test_host_ops test_tracker_flow"
     exit 0
fi

# fixed binary name (run.sh calls ./TrackerV2), independent of the folder name
name=TrackerV2
if [[ "$CXX"  == *"sysroot"* ]];then
$CXX -O2 -fno-inline -I. \
     -I=/install/Debug/include \
     -I=/install/Release/include \
     -L=/install/Debug/lib \
     -L=/install/Release/lib \
     -I$PWD/src/common -o $name -std=c++17 \
     -I$PWD/src/DPU -o $name -std=c++17 \
     -I$PWD/src/tracker -o $name -std=c++17 \
     $PWD/src/main.cpp \
     $PWD/src/common/common.cpp  \
     $PWD/src/DPU/dpu.cpp  \
     $PWD/src/DPU/model_runner.cpp  \
     $PWD/src/tracker/tracker.cpp  \
     $PWD/src/tracker/AItracker.cpp  \
     $PWD/src/tracker/lightfc_ops.cpp  \
     $PWD/src/tracker/lightfc_config.cpp  \
     -Wl,-rpath=$PWD/lib \
     -lvart-runner \
     ${OPENCV_FLAGS} \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread
else
$CXX -O2 -fno-inline -I. \
     -I${install_prefix_default}.Debug/include \
     -I${install_prefix_default}.Release/include \
     -L${install_prefix_default}.Debug/lib \
     -L${install_prefix_default}.Release/lib \
     -Wl,-rpath=${install_prefix_default}.Debug/lib \
     -Wl,-rpath=${install_prefix_default}.Release/lib \
     -I$PWD/src/common -o $name -std=c++17 \
     -I$PWD/src/DPU -o $name -std=c++17 \
     -I$PWD/src/tracker -o $name -std=c++17 \
     $PWD/src/main.cpp \
     $PWD/src/common/common.cpp  \
     $PWD/src/DPU/dpu.cpp  \
     $PWD/src/DPU/model_runner.cpp  \
     $PWD/src/tracker/tracker.cpp  \
     $PWD/src/tracker/AItracker.cpp  \
     $PWD/src/tracker/lightfc_ops.cpp  \
     $PWD/src/tracker/lightfc_config.cpp  \
     -Wl,-rpath=$PWD/lib \
     -lvart-runner \
     ${OPENCV_FLAGS} \
     -lopencv_videoio  \
     -lopencv_imgcodecs \
     -lopencv_highgui \
     -lopencv_imgproc \
     -lopencv_core \
     -lglog \
     -lxir \
     -lunilog \
     -lpthread
fi
