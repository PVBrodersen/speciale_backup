# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2023 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
# SaveToAvi.py shows how to create an AVI video from a vector of
# images. It relies on information provided in the Enumeration, Acquisition,
# and NodeMapInfo examples.
#
# This example introduces the SpinVideo class, which is used to quickly and
# easily create various types of AVI videos. It demonstrates the creation of
# three types: uncompressed, MJPG, and H264.
#
# Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
# More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
# Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics

import PySpin
import sys


class AviType:
    """'Enum' to select AVI video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264 = 2

chosenAviType = AviType.H264  # change me!



def save_list_to_avi(nodemap, nodemap_tldevice, images, framerate, filename):
    """
    This function prepares, saves, and cleans up an AVI video from a vector of images.

    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param images: List of images to save to an AVI video.
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type images: list of ImagePtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** CREATING VIDEO ***')

    try:
        result = True

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        # if PySpin.IsReadable(node_serial):
        device_serial_number = node_serial.GetValue()
        #     print('Device serial number retrieved as %s...' % device_serial_number)

        # # Get the current frame rate; acquisition frame rate recorded in hertz
        # #
        # # *** NOTES ***
        # # The video frame rate can be set to anything; however, in order to
        # # have videos play in real-time, the acquisition frame rate can be
        # # retrieved from the camera.
        # node_acquisition_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        # node_acquisition_enable.SetValue(True)
        # node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))

        # if not PySpin.IsAvailable(node_acquisition_framerate):
        #     print('Frame rate node unavailable. Aborting...')

        # if not PySpin.IsReadable(node_acquisition_framerate):
        #     print('Unable to retrieve frame rate. Aborting...')
        #     return False
        
        # framerate_enable = node_acquisition_enable.GetValue()
        # node_acquisition_framerate.SetValue(framerate)
        # framerate_to_set = node_acquisition_framerate.GetValue()

        # print(f"Is enabled: {framerate_enable} and framerate: {framerate_to_set}")
        # Select option and open AVI filetype with unique filename
        #
        # *** NOTES ***
        # Depending on the filetype, a number of settings need to be set in
        # an object called an option. An uncompressed option only needs to
        # have the video frame rate set whereas videos with MJPG or H264
        # compressions should have more values set.
        #
        # Once the desired option object is configured, open the AVI file
        # with the option in order to create the image file.
        #
        # Note that the filename does not need to be appended to the
        # name of the file. This is because the AVI recorder object takes care
        # of the file extension automatically.
        #
        # *** LATER ***
        # Once all images have been added, it is important to close the file -
        # this is similar to many other standard file streams.

        avi_recorder = PySpin.SpinVideo()

        if chosenAviType == AviType.UNCOMPRESSED:
            avi_filename = filename + device_serial_number

            option = PySpin.AVIOption()
            option.frameRate = framerate
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        elif chosenAviType == AviType.MJPG:
            avi_filename = filename + device_serial_number

            option = PySpin.MJPGOption()
            option.frameRate = framerate
            option.quality = 75
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        elif chosenAviType == AviType.H264:
            avi_filename = filename + device_serial_number

            option = PySpin.H264Option()
            option.frameRate = framerate
            option.bitrate = 1000000
            option.height = images[0].GetHeight()
            option.width = images[0].GetWidth()

        else:
            print('Error: Unknown AviType. Aborting...')
            return False

        avi_recorder.Open(avi_filename, option)

        # Construct and save AVI video
        #
        # *** NOTES ***
        # Although the video file has been opened, images must be individually
        # appended in order to construct the video.
        print('Appending %d images to AVI file: %s.avi...' % (len(images), avi_filename))

        for i in range(len(images)):
            avi_recorder.Append(images[i])
            print('Appended image %d...' % i)

        # Close AVI file
        #
        # *** NOTES ***
        # Once all images have been appended, it is important to close the
        # AVI file. Notice that once an AVI file has been closed, no more
        # images can be added.

        avi_recorder.Close()
        print('Video saved at %s.avi' % avi_filename)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('\n*** DEVICE INFORMATION ***\n')

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not readable.')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result


def acquire_images(cam):
    """
    This function acquires 30 images from a device, stores them in a list, and returns the list.
    please see the Acquisition example for more in-depth comments on acquiring images.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** IMAGE ACQUISITION ***\n')
    try:
        result = True

        # # Set acquisition mode to continuous
        # node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
        # if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
        #     print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
        #     return False

        # # Retrieve entry node from enumeration node
        # node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        # if not PySpin.IsReadable(node_acquisition_mode_continuous):
        #     print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
        #     return False

        # acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

        # node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        # print('Acquisition mode set to continuous...')

        # #  Begin acquiring images
        # cam.BeginAcquisition()

        # print('Acquiring images...')

        # # Retrieve, convert, and save images
        # #images = list()

        # Create ImageProcessor instance for post processing images
        processor = PySpin.ImageProcessor()

        # Set default image processor color processing method
        #
        # *** NOTES ***
        # By default, if no specific color processing algorithm is set, the image
        # processor will default to NEAREST_NEIGHBOR method.
        processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

        
        try:
            #  Retrieve next received image
            image_result = cam.GetNextImage(1000)

            #  Ensure image completion
            if image_result.IsIncomplete():
                print('Image incomplete with image status %d...' % image_result.GetImageStatus())
                out_img = processor.Convert(image_result, PySpin.PixelFormat_Mono8)
            else:
                #  Print image information; height and width recorded in pixels
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                #print('Grabbed Image, width = %d, height = %d' % (width, height))

                #  Convert image to mono 8
                out_img = processor.Convert(image_result, PySpin.PixelFormat_Mono8)

                #  Release image
                image_result.Release()
                print('')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        # End acquisition
        #cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result, out_img


def run_single_camera(cam):
    """
    This function acts as the body of the example; please see NodeMapInfo example
    for more in-depth comments on setting up cameras.

    :param cam: Camera to run example on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Acquire list of images
        err, images = acquire_images(cam, nodemap)
        if err < 0:
            return err

        result &= save_list_to_avi(nodemap, nodemap_tldevice, images)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


