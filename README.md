# Applevision Kalman Filter

This project is a [ROS](https://www.ros.org/) package that contains a python service which will perform kalman filter calculations given sensor and control inputs. To use this package you'll need to install ROS melodic and setup a workspace for development: more info on that in the [ROS tutorials](http://wiki.ros.org/ROS/Installation).

## Block verification tests

To verify this individual block, simply run the provided test file with python. If no exceptions occur, the block verification succeded.
```sh
python3 test/block_verification.py
```