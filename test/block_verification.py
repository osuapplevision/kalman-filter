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
        # setup the mock distance sensor (pubs to topic appl_dist)
        mock_pub = Mock()
        mock_sub = Mock()
        mock_rospy.Publisher.return_value = mock_pub
        mock_rospy.Subscriber.return_value = mock_sub

        # setup the mock computer vision (writes coordinates to a file)
        coordpath = Path(dir).joinpath('coords.txt')
        coordpath.write_text('0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')

        # setup the kalman filter module
        applevision_kalman_run.main(coordpath)
        assert mock_rospy.Subscriber.call_args[0][0] \
            == 'appl_dist'  # verify subscribe topic
        invoke_handler = mock_rospy.Subscriber.call_args[0][2]

        # invoke the kalman filter by fake publishing
        data = Mock()
        data.data = 1000
        invoke_handler(data)

        # test that the output was published to the output topic
        mock_pub.publish.assert_called_once_with(data=[
            0.0,
            0.0,
            1000.0,  # x, y, z (mm)
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            97560.97601475925  # varience
        ])


@patch('applevision_kalman_run.rospy')
def test_block_througput(mock_rospy):
    COUNT = 10000
    with TemporaryDirectory() as dir:
        coordpath = Path(dir).joinpath('coords.txt')
        coordpath.write_text('0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')
        applevision_kalman_run.main(coordpath)
        invoke_handler = mock_rospy.Subscriber.call_args[0][2]

        data = Mock()
        data.data = 100

        # fake publish COUNT times
        start = time.time()
        for _ in range(COUNT):
            invoke_handler(data)
        end = time.time()

        # calculate throughput
        ops_sec = COUNT / (end - start)
        print(f'{ops_sec:.2f} ops/sec')

        # ensure we can operate >30Hz
        assert ops_sec > 30


if __name__ == '__main__':
    test_block_io()
    test_block_througput()

    print('All tests passed!')
