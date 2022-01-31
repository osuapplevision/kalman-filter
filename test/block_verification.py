import sys
from pathlib import Path
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory
import time

PACKAGE_DIR = Path(__file__).parent.parent
sys.path.append(str(PACKAGE_DIR.joinpath('src')))
sys.path.append(str(PACKAGE_DIR.joinpath('bin')))

import applevision_kalman_run


@patch('applevision_kalman_run.rospy')
def test_block_io(mock_rospy):
    with TemporaryDirectory() as dir:
        mock_pub = Mock()
        mock_sub = Mock()
        mock_rospy.Publisher.return_value = mock_pub
        mock_rospy.Subscriber.return_value = mock_sub
        coordpath = Path(dir).joinpath('coords.txt')
        coordpath.write_text('0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')
        applevision_kalman_run.main(coordpath)
        invoke_handler = mock_rospy.Subscriber.call_args[0][2]

        # test publish
        data = Mock()
        data.data = 0
        invoke_handler(data)

        # test that the output was published
        mock_pub.publish.assert_called_once_with(data=[
            0.0, 0.0, 0.5417184839230509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.08667495742768817
        ])


@patch('applevision_kalman_run.rospy')
def test_block_througput(mock_rospy):
    COUNT = 10000
    with TemporaryDirectory() as dir:
        coordpath = Path(dir).joinpath('coords.txt')
        coordpath.write_text('0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')
        applevision_kalman_run.main(coordpath)
        invoke_handler = mock_rospy.Subscriber.call_args[0][2]

        # test publish
        data = Mock()
        data.data = 1

        start = time.time()
        for _ in range(COUNT):
            invoke_handler(data)
        end = time.time()
        ops_sec = COUNT / (end - start)

        print(f'{ops_sec:.2f} ops/sec')

        assert ops_sec > 30


if __name__ == '__main__':
    test_block_througput()