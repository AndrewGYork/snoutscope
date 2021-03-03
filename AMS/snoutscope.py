# Imports from the python standard library:
import time
import os
import atexit
import threading
import queue

# Third party imports, installable via pip:
import numpy as np
from scipy.ndimage import zoom, rotate
from tifffile import imread, imwrite
# import matplotlib
# We only import matplotlib if/when we call Snoutscope._plot_voltages()

# Our stuff, from github.com/AndrewGYork/tools. Don't pip install.
# One .py file per module, copy files to your local directory.
import pco # Install PCO's SDK to get relevant DLLs
import ni # Install NI-DAQmx to get relevant DLLs
import sutter
import physik_instrumente as p_i
import thorlabs
import proxy_objects
from proxied_napari import display

# Microscope configuration (edit as needed):
M1 = 200 / 2; Mscan = 70 / 70; M2 = 5 / 357; M3 = 200 / 5
MRR = M1 * Mscan * M2; Mtot = MRR * M3;
camera_px_um = 6.5; sample_px_um = camera_px_um / Mtot
tilt = np.deg2rad(30)

class Snoutscope:
    def __init__(
        self,
        bytes_per_data_buffer,
        num_data_buffers,
        bytes_per_preview_buffer,
        ):
        """
        We use bytes_per_buffer and bytes_per_preview_buffer to specify
        the shared_memory_sizes for the child processes.
        """
        # TODO: Think about an elegant solution for printing and threading
        self._init_shared_memory(
            bytes_per_data_buffer, num_data_buffers, bytes_per_preview_buffer)
        self.unfinished_tasks = queue.Queue()
        slow_fw_init = threading.Thread(target=self._init_filter_wheel) #~5.3s
        slow_camera_init = threading.Thread(target=self._init_camera) #~3.6s
        slow_snoutfocus_init = threading.Thread(target=self._init_snoutfocus)#1s
        slow_focus_init = threading.Thread(target=self._init_focus_piezo) #~0.6s
        slow_stage_init = threading.Thread(target=self._init_XY_stage) #~0.4s
        slow_fw_init.start()
        slow_camera_init.start()
        slow_snoutfocus_init.start()
        slow_focus_init.start()
        slow_stage_init.start()
        self._init_display() #~1.3s
        self._init_preprocessor() #~0.8s
        self._init_ao() #~0.2s
        slow_stage_init.join()
        slow_focus_init.join()
        slow_snoutfocus_init.join()
        slow_camera_init.join()
        slow_fw_init.join()
        print('Finished initializing Snoutscope')
        # Note: you still have to call .apply_settings() before you can .snap()

    def apply_settings(
        self,
        roi=None, # Dict, see pco.py ._set_roi()
        shear_px_per_scan_step=None, # Int (or Float but be careful!)
        illumination_time_microseconds=None, # Float
        volumes_per_buffer=None, # Int
        slices_per_volume=None,  # Int
        channels_per_slice=None, # Tuple of strings
        power_per_channel=None,  # Tuple of floats
        filter_wheel_position=None, # Int
        focus_piezo_position_um=None, # Float or (Float, "relative")
        XY_stage_position_mm=None, # (Float, Float, optional: "relative")
        timestamp_mode=None, # String, see pco.py ._set_timestamp_mode()
        ):
        ''' Apply any settings to the scope that need to be configured before
            acquring an image. Think filter wheel position, stage position,
            camera settings, light intensity, calculate daq voltages, etc.'''
        args = locals()
        args.pop('self')
        def settings_task(custody):
            custody.switch_from(None, to=self.camera)
            # We own the camera, safe to change settings
            self._settings_are_sane = False # In case the thread crashes
            # Attributes must be set previously or currently:
            for k, v in args.items(): 
                if v is not None:
                    setattr(self, k, v) # A lot like self.x = x
                assert hasattr(self, k), (
                    'Attribute %s must be set by apply_settings()'%k)
            # Send hardware commands, slowest to fastest:
            if XY_stage_position_mm is not None:
                x, y = XY_stage_position_mm[0:2]
                # Absolute position, or relative motion?
                if len(XY_stage_position_mm) > 2:
                    move_type = XY_stage_position_mm[2]
                    assert move_type == "relative"
                    x0, y0 = self.XY_stage.get_position()
                    x, y = x0+x, y0+y
                self.XY_stage.move(x, y, blocking=False)
            if filter_wheel_position is not None:
                self.filter_wheel.move(filter_wheel_position,
                                       speed=6,
                                       block=False)
            if focus_piezo_position_um is not None:
                try: # Absolute position?
                    z = float(focus_piezo_position_um)
                except TypeError: # Relative position?
                    assert focus_piezo_position_um[1] == "relative"
                    z = (float(focus_piezo_position_um[0]) +
                         self.focus_piezo.get_real_position())
                self.focus_piezo.move(z) # nonblocking
            if (roi is not None or
                illumination_time_microseconds is not None or
                timestamp_mode is not None):
                self.camera.disarm()
                if roi is not None: self.camera._set_roi(roi)
                if (roi, illumination_time_microseconds) != (None, None):
                    self.camera._set_exposure_time(
                        self.illumination_time_microseconds +
                        self.camera.rolling_time_microseconds)
                if timestamp_mode is not None:
                    self.camera._set_timestamp_mode(timestamp_mode)
                self.camera.arm(16)
            self._calculate_voltages()
            # Finalize hardware commands, fastest to slowest:
            if focus_piezo_position_um is not None:
                self.focus_piezo._finish_moving()
            if filter_wheel_position is not None:
                self.filter_wheel._finish_moving()
            if XY_stage_position_mm is not None:
                self.XY_stage.finish_moving()
            self._settings_are_sane = True
            custody.switch_from(self.camera, to=None)
        settings_thread = proxy_objects.launch_custody_thread(
            target=settings_task, first_resource=self.camera)
        self.unfinished_tasks.put(settings_thread)
        return settings_thread

    def snap(self, display=True, filename=None, delay_seconds=None):
        if delay_seconds is not None and delay_seconds > 3:
            self.snoutfocus(delay_seconds=delay_seconds) # Might as well!
            delay_seconds = None
        def snap_task(custody):
            custody.switch_from(None, to=self.camera)
            if delay_seconds is not None:
                time.sleep(delay_seconds) # simple but not precise
            assert hasattr(self, '_settings_are_sane'), (
                'Please call .apply_settings() before using .snap()')
            assert self._settings_are_sane, (
                'Did .apply_settings() fail? Please call it again, ' +
                'with all arguments specified.')
            exposures_per_buffer = (len(self.channels_per_slice) *
                                    self.slices_per_volume *
                                    self.volumes_per_buffer)
            data_buffer = self._get_data_buffer(
                (exposures_per_buffer, self.camera.height, self.camera.width),
                'uint16')
            # camera.record_to_memory() blocks, so we use a thread:
            camera_thread = threading.Thread(
                target=self.camera.record_to_memory,
                args=(exposures_per_buffer,),
                kwargs={'out': data_buffer,
                        'first_trigger_timeout_seconds': 10},)
            camera_thread.start()
            # There's a race here. The PCO camera starts with N empty
            # single-frame buffers (typically 16), which are filled by
            # the triggers sent by ao.play_voltages(). The camera_thread
            # empties them, hopefully fast enough that we never run out.
            # So far, the camera_thread seems to both start on time, and
            # keep up reliably once it starts, but this could be
            # fragile.
            self.ao.play_voltages(self.voltages, block=False)
            ## TODO: consider finished playing all voltages before moving on...
            camera_thread.join()
            # Acquisition is 3D, but display and filesaving are 5D:
            data_buffer = data_buffer.reshape(self.volumes_per_buffer,
                                              self.slices_per_volume,
                                              len(self.channels_per_slice),
                                              data_buffer.shape[-2],
                                              data_buffer.shape[-1])
            if display:
                # We have custody of the camera so attribute access is safe:
                scan_step_size_um = self.scan_step_size_um
                skip_rows = 0
                if self.timestamp_mode != "off":
                    skip_rows = 8 # Ignore rows so timestamps don't dominate
                preview_me = data_buffer[:, :, :, skip_rows:, :]
                prev_shape = (
                    (self.volumes_per_buffer,
                     len(self.channels_per_slice),) +
                    Preprocessor.three_traditional_projections_shape(
                        self.slices_per_volume,
                        preview_me.shape[-2],
                        preview_me.shape[-1],
                        scan_step_size_um))
                custody.switch_from(self.camera, to=self.preprocessor)
                preview_buffer = self._get_preview_buffer(prev_shape, 'uint16')
                for vo in range(preview_me.shape[0]):
                    for ch in range(preview_me.shape[2]):
                        self.preprocessor.three_traditional_projections(
                            data_buffer[vo, :, ch, skip_rows:, :],
                            scan_step_size_um,
                            out=preview_buffer[vo, ch, :, :])
                custody.switch_from(self.preprocessor, to=self.display)
                self.display.show_image(preview_buffer)
                custody.switch_from(self.display, to=None)
                if filename is not None:
                    root, ext = os.path.splitext(filename)
                    preview_filename = root + '_preview' + ext
                    print("Saving file", preview_filename, end=' ')
                    imwrite(preview_filename, preview_buffer, imagej=True)
                    print("done.")
                self._release_preview_buffer(preview_buffer)
            else:
                custody.switch_from(self.camera, to=None)
            # TODO: if file saving turns out to disrupt other activities
            # in the main process, make a FileSaving proxy object.
            if filename is not None:
                print("Saving file", filename, end=' ')
                imwrite(filename, data_buffer, imagej=True)
                print("done.")
            self._release_data_buffer(data_buffer)
        snap_thread = proxy_objects.launch_custody_thread(
            target=snap_task, first_resource=self.camera)
        self.unfinished_tasks.put(snap_thread)
        return snap_thread

    def snoutfocus(self, filename=None, delay_seconds=None):
        def snoutfocus_task(custody):
            custody.switch_from(None, to=self.camera) # safe to change settings
            if delay_seconds is not None:
                start_time = time.perf_counter()
                if delay_seconds > 3: # 3 seconds is def. enough time to focus
                    time.sleep(delay_seconds - 3)
            self._settings_are_sane = False # In case the thread crashes
            # Record the settings we'll have to reset:
            old_filter_wheel_position = self.filter_wheel_position
            old_roi = self.camera.roi
            old_exp_us = self.camera.exposure_time_microseconds
            timestamp_mode = self.camera.timestamp_mode
            # Get microscope settings ready to take our measurement:
            self.filter_wheel.move(1, speed=6, block=False) # Empty slot
            self.snoutfocus_controller.set_voltage(
                0, block=False) # Slow, but filter wheel is slower
            self.camera.disarm()
            self.camera._set_roi({'left': 901, 'top': 901, # reduce for speed
                                  'right': 1160, 'bottom': 1148})
            self.camera._set_exposure_time(100) # Microseconds
            self.camera.arm(16)
            # Calculate voltages for the analog-out card:
            exp_pix = self.ao.s2p(1e-6*self.camera.exposure_time_microseconds)
            roll_pix = self.ao.s2p(1e-6*self.camera.rolling_time_microseconds)
            jitter_pix = max(self.ao.s2p(29e-6), 1) # Maybe as low as 27 us?
            piezo_settling_pix = self.ao.s2p(0.000) # Not yet measured
            period_pix = (max(exp_pix, roll_pix, piezo_settling_pix)
                          + jitter_pix)
            piezo_voltage_limit = 150 # 15 um for current piezo
            piezo_voltage_step = 2 # 200 nm steps
            n2c = self.names_to_voltage_channels # A temporary nickname
            voltages = []
            for piezo_voltage in np.arange(
                0, piezo_voltage_limit+piezo_voltage_step, piezo_voltage_step):
                v = np.zeros((period_pix, self.ao.num_channels), 'float64')
                v[:roll_pix, n2c['camera']] = 5
                v[:, n2c['snoutfocus_piezo']] = 10*(piezo_voltage /
                                                    piezo_voltage_limit) # 10 V
                voltages.append(v)
            exposures_per_buffer = len(voltages)
            voltages = np.concatenate(voltages, axis=0)
            # Allocate memory and finalize microscope settings:
            data_buffer = self._get_data_buffer(
                (exposures_per_buffer, self.camera.height, self.camera.width),
                'uint16') # TODO: effectively a min size for data buffers!
            self.snoutfocus_controller._finish_set_voltage()
            self.filter_wheel._finish_moving()
            # Take pictures while moving the snoutfocus piezo:
            camera_thread = threading.Thread(
                target=self.camera.record_to_memory,
                args=(exposures_per_buffer,),
                kwargs={'out': data_buffer,
                        'first_trigger_timeout_seconds': 10},)
            camera_thread.start()
            self.ao.play_voltages(voltages, block=False) # Ends at 0 V
            camera_thread.join()
            # Start cleaning up after ourselves:
            self.filter_wheel.move(old_filter_wheel_position,
                                   speed=6, block=False)
            # Inspect the images to find/set best snoutfocus piezo position:
            inspect_me = data_buffer[:, 8:, :] # Top rows might have timestamps
            if np.max(inspect_me) < 5 * np.min(inspect_me):
                print('WARNING: snoutfocus laser intensity is low:',
                      'is the laser/shutter powered up?')
            controller_voltage = piezo_voltage_step * np.unravel_index(
                np.argmax(inspect_me), inspect_me.shape)[0]
            if (controller_voltage == 0 or
                controller_voltage == piezo_voltage_limit):
                print('WARNING: snoutfocus piezo is out of range!')
            self.snoutfocus_controller.set_voltage(
                controller_voltage, block=False)
            print('Snoutfocus piezo voltage =', controller_voltage)
            # Finish cleaning up after ourselves:
            self.camera.disarm()
            self.camera._set_roi(old_roi)
            self.camera._set_exposure_time(old_exp_us)
            self.camera.arm(16)
            self.snoutfocus_controller._finish_set_voltage()
            self.filter_wheel._finish_moving()
            self._settings_are_sane = True
            # We might want to hold camera custody for a fixed amount of time:
            if delay_seconds is not None:
                while time.perf_counter() - start_time < delay_seconds:
                    time.sleep(0.001)
            custody.switch_from(self.camera, to=None)
            if filename is not None:
                imwrite(filename, data_buffer[:, np.newaxis, :, :])
            self._release_data_buffer(data_buffer)
        snoutfocus_thread = proxy_objects.launch_custody_thread(
            target=snoutfocus_task, first_resource=self.camera)
        self.unfinished_tasks.put(snoutfocus_thread)
        return snoutfocus_thread

    def spiral_tiling_preview(
        self,
        num_spirals=1,
        dx_mm=0.1,
        dy_mm=0.1,
        ):
        ####################
        # WORK IN PROGRESS #
        ####################
        # TODO: xy, or yx? What's stage leftright vs. screen leftright?
        # If we get this straight, this code will suck less.
## TODO: - make sure stage is at max velocity and then return
        assert dx_mm and dy_mm < 1
        assert num_spirals < 4 # C'mon don't be silly, this is still a lot
        data_filename = "spiral_%02i_%02i.tif"
        preview_filename = "spiral_%02i_%02i_preview.tif"
        ix, iy = [0], [0] # Mutable, integer coords relative to starting tile
        snap_tasks = []
        def move(x, y):
            ix[0], iy[0] = ix[0]+x, iy[0]+y
            self.apply_settings(
                XY_stage_position_mm=(dx_mm*x, dy_mm*y, "relative"))
            snap_tasks.append(self.snap(
                filename=data_filename%(ix[0]+num_spirals, iy[0]+num_spirals),
                display=True))
        def move_up():    move( 0,  1)
        def move_down():  move( 0, -1)
        def move_left():  move(-1,  0)
        def move_right(): move( 1,  0)
        move(0, 0) # get the central tile
        for which_spiral in range(num_spirals): # This took me a while...
            for i in range(1):
                move_up()
            for i in range(1 + 2*which_spiral):
                move_right()
            for i in range(2 + 2*which_spiral):
                move_down()
            for i in range(2 + 2*which_spiral):
                move_left()
            for i in range(2 + 2*which_spiral):
                move_up()
        self.apply_settings( # Return to original position
            XY_stage_position_mm=(-dx_mm*ix[0], -dy_mm*iy[0], "relative"))
        def display_preview_task(custody):
            # To preserve order we follow the same custody pattern as snap():
            custody.switch_from(None, to=self.camera)
            custody.switch_from(self.camera, to=self.preprocessor)
            custody.switch_from(self.preprocessor, to=self.display)
            # You can't display until the preview files have saved to disk:
            for st in snap_tasks: st.join()
            # Now you can start to reload the preview images:
            # TODO: re-try (with small delay) if file not found yet?
            first_tile = imread(preview_filename%(0, 0))
            ud_pix, lr_pix = first_tile.shape[-2:]
            preview_shape = list(first_tile.shape)
            preview_shape[-2] = ud_pix * (1 + 2*num_spirals)
            preview_shape[-1] = lr_pix * (1 + 2*num_spirals)
            # We're ready to hog some shared resources:
            # TODO: seriously consider adding spiral_preview_buffers to avoid
            # asking for memory that's not available
            preview_buffer = self._get_preview_buffer(preview_shape, 'uint16')
            preview_buffer[..., :ud_pix, :lr_pix] = first_tile
            for i in range(1 + 2*num_spirals):
                for j in range(1 + 2*num_spirals):
                    if (i, j) == (0, 0): continue # We already did this tile
                    preview_buffer[...,
                                   ud_pix*j:ud_pix*(j+1),
                                   lr_pix*i:lr_pix*(i+1)
                                   ] = imread(preview_filename%(i, j))
            self.display.show_image(preview_buffer)
            custody.switch_from(self.display, to=None)
            self._release_preview_buffer(preview_buffer)
        display_thread = proxy_objects.launch_custody_thread(
            target=display_preview_task, first_resource=self.camera)
        self.unfinished_tasks.put(display_thread)

    def finish_all_tasks(self):
        collected_tasks = []
        while True:
            try:
                th = self.unfinished_tasks.get_nowait()
            except queue.Empty:
                break
            th.join()
            collected_tasks.append(th)
        return collected_tasks

    def close(self, finish_all_tasks=True):
        if finish_all_tasks:
            self.finish_all_tasks()
        self.ao.close()
        self.filter_wheel.close()
        self.camera.close()
        self.snoutfocus_controller.close()
        self.focus_piezo.close()
        self.XY_stage.close()
        self.display.close()
        print('Closed Snoutscope')

    def _init_shared_memory(
        self,
        bytes_per_data_buffer,
        num_data_buffers,
        bytes_per_preview_buffer,
        ):
        """
        Each buffer is acquired in deterministic time with a single play
        of the ao card.
        """
        num_preview_buffers = 3 # 3 for preprocess, display and filesave
        assert bytes_per_data_buffer > 0 and num_data_buffers > 0
        assert bytes_per_preview_buffer > 0
        print("Allocating shared memory...", end=' ')
        self.pm = proxy_objects.ProxyManager(shared_memory_sizes=(
            (bytes_per_data_buffer,   ) * num_data_buffers +
            (bytes_per_preview_buffer,) * num_preview_buffers))
        print("done allocating memory.")
        self.data_buffer_queue = queue.Queue(maxsize=num_data_buffers)
        for i in range(num_data_buffers):
            self.data_buffer_queue.put(i)
        self.preview_buffer_queue = queue.Queue(maxsize=num_preview_buffers)
        for i in range(num_preview_buffers):
            self.preview_buffer_queue.put(i + num_data_buffers) # pointer math!

    def _init_ao(self):
        self.names_to_voltage_channels = {
            'camera': 0,
            'galvo': 4,
            'snoutfocus_piezo': 6,
            'LED_power': 12, 
            '405_TTL': 16,
            '405_power': 17,
            '488_TTL': 20,
            '488_power': 21,
            '561_TTL': 24,
            '561_power': 25,
            '640_TTL': 28,
            '640_power': 29,}
        print("Initializing ao card...", end=' ')
        self.ao = ni.Analog_Out(num_channels=30,
                                rate=1e5,
                                daq_type='6739',
                                board_name='PXI1Slot2',
                                verbose=False)
        print("done with ao.")
        atexit.register(self.ao.close)

    def _init_filter_wheel(self):
        print("Initializing filter wheel...")
        self.filter_wheel = sutter.Lambda_10_3(which_port='COM3', verbose=False)
        print("done with filter wheel.")
        self.filter_wheel_position = 0
        atexit.register(self.filter_wheel.close)

    def _init_camera(self):
        print("Initializing camera... (this sometimes hangs)")
        self.camera = self.pm.proxy_object(pco.Camera, verbose=False,
                                           close_method_name='close')
        self.camera.apply_settings(trigger='external_trigger')
        print("done with camera.")

    def _init_snoutfocus(self):
        print("Initializing snoutfocus piezo...")
        self.snoutfocus_controller = thorlabs.MDT694B_piezo_controller(
            which_port='COM7', verbose=False)
        print("done with snoutfocus piezo.")
        atexit.register(self.snoutfocus_controller.close)

    def _init_focus_piezo(self):
        print("Initializing focus piezo...")
        self.focus_piezo = p_i.E753_Z_Piezo(which_port='COM6', verbose=False)
        print("done with focus piezo.")
        atexit.register(self.focus_piezo.close)

    def _init_XY_stage(self):
        print("Initializing XY stage...")
        self.XY_stage = p_i.C867_XY_Stage(which_port='COM5', verbose=False)
        print("done with XY stage.")
        atexit.register(self.XY_stage.close)

    def _init_preprocessor(self):
        print("Initializing preprocessor...")
        self.preprocessor = self.pm.proxy_object(Preprocessor)
        print("done with preprocessor.")

    def _init_display(self):
        print("Initializing display...")
        self.display = display(proxy_manager=self.pm)
        print("done with display.")

    def _calculate_voltages(self):
        # We'll use this a lot, so a short nickname is nice:
        n2c = self.names_to_voltage_channels
        # Input sanitization
        illumination_sources = ('LED', '405', '488', '561', '640',
                                '405_on_during_rolling')
        for channel in self.channels_per_slice:
            assert channel in illumination_sources
        assert len(self.channels_per_slice) > 0
        assert len(self.power_per_channel) == len(self.channels_per_slice)
        for power in self.power_per_channel:
            assert 0 <= power <= 100
        assert self.slices_per_volume == int(self.slices_per_volume)
        assert self.slices_per_volume > 0
        assert self.volumes_per_buffer == int(self.volumes_per_buffer)
        assert self.volumes_per_buffer > 0

        # Timing information
        exposure_pix = self.ao.s2p(1e-6*self.camera.exposure_time_microseconds)
        rolling_pix = self.ao.s2p(1e-6*self.camera.rolling_time_microseconds)

        jitter_pix = max(self.ao.s2p(29e-6), 1) # Maybe as low as 27 us?
        period_pix = max(exposure_pix, rolling_pix) + jitter_pix

        # Calculate galvo voltages from volume settings:
        self.voxel_aspect_ratio = self.shear_px_per_scan_step * np.tan(tilt)
        self.scan_step_size_um = (
            self.voxel_aspect_ratio * sample_px_um / np.sin(tilt))
        
        scan_range_um = self.scan_step_size_um * (self.slices_per_volume - 1)
        assert 0 <= scan_range_um <= 201 # optical limit
        galvo_volts_per_um = 4.5 / 110 # calibrated using graticule
        galvo_scan_volts = galvo_volts_per_um * scan_range_um
        galvo_voltages = np.linspace(-galvo_scan_volts/2,
                                     galvo_scan_volts/2,
                                     self.slices_per_volume)

        # Calculate voltages
        voltages = []
        for vo in range(self.volumes_per_buffer):
            # TODO: either bidirectional volumes, or smoother galvo flyback
            for sl in range(self.slices_per_volume):
                for ch, pw in zip(self.channels_per_slice,
                                  self.power_per_channel):
                    v = np.zeros((period_pix, self.ao.num_channels), 'float64')
                    # Camera trigger:
                    v[:rolling_pix, n2c['camera']] = 5 # falling edge->light on!
                    # Galvo step:
                    v[:, n2c['galvo']] = galvo_voltages[sl] # galvo
                    if channel in ('405_on_during_rolling',):
                        light_on_pix = 0
                    else:
                        light_on_pix = rolling_pix
                    # Illumination TTL trigger:
                    if ch != 'LED': # i.e. the laser channels
                        v[light_on_pix:period_pix - jitter_pix,
                          n2c[ch + '_TTL']] = 3
                    # Illumination power modulation:
                    v[light_on_pix:period_pix - jitter_pix,
                      n2c[ch + '_power']] = 4.5 * pw / 100
                    voltages.append(v)
        self.voltages = np.concatenate(voltages, axis=0)

    def _plot_voltages(self):
        import matplotlib.pyplot as plt
        # Reverse lookup table; channel numbers to names:
        c2n = {v:k for k, v in self.names_to_voltage_channels.items()}
        for c in range(self.voltages.shape[1]):
            plt.plot(self.voltages[:, c], label=c2n.get(c, f'ao-{c}'))
        plt.legend(loc='upper right')
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [self.ao.p2s(l) for l in xlocs])
        plt.ylabel('Volts')
        plt.xlabel('Seconds')
        plt.show()

    def _get_data_buffer(self, shape, dtype):
        which_mp_array = self.data_buffer_queue.get()
        try:
            data_buffer = self.pm.shared_numpy_array(
                which_mp_array, shape, dtype)
        except ValueError as e:
            print("Your Snoutscope data buffers are too small to hold a",
                  shape, "array of type", dtype)
            print("Either ask for a smaller array, or make a new Snoutscope",
                  " object with more 'bytes_per_data_buffer'.")
            raise e
        return data_buffer

    def _release_data_buffer(self, shared_numpy_array):
        assert isinstance(shared_numpy_array, proxy_objects._SharedNumpyArray)
        which_mp_array = shared_numpy_array.buffer
        self.data_buffer_queue.put(which_mp_array)

    def _get_preview_buffer(self, shape, dtype):
        which_mp_array = self.preview_buffer_queue.get()
        try:
            preview_buffer = self.pm.shared_numpy_array(
                which_mp_array, shape, dtype)
        except ValueError as e:
            print("Your Snoutscope preview buffers are too small to hold a",
                  shape, "array of type", dtype)
            print("Either ask for a smaller array, or make a new Snoutscope",
                  " object with more 'bytes_per_preview_buffer'.")
            raise e
        return preview_buffer

    def _release_preview_buffer(self, shared_numpy_array):
        assert isinstance(shared_numpy_array, proxy_objects._SharedNumpyArray)
        which_mp_array = shared_numpy_array.buffer
        self.preview_buffer_queue.put(which_mp_array)


def legalize_voxel_aspect_ratio(aspect_ratio):
    return max(int(round(aspect_ratio / np.tan(tilt))), 1) * np.tan(tilt)

def legalize_scan(aspect_ratio, scan_range_um):
    aspect_ratio = legalize_voxel_aspect_ratio(aspect_ratio)
    scan_step_size_um = aspect_ratio * sample_px_um / np.sin(tilt)
    slices_per_vol = 1 + int(round(scan_range_um / scan_step_size_um))
    scan_range_um = scan_step_size_um * (slices_per_vol - 1)
    assert 0 <= scan_range_um <= 201 # optical limit
    return slices_per_vol, scan_step_size_um # watch out for fencepost errors!

class Preprocessor:
    @staticmethod
    def three_traditional_projections_shape(
        scan_steps,
        prop_pxls,
        width_pxls,
        scan_step_size_um,
        separation_line_px_width=10,
        ):
        # Calculate max pixel shift for shearing on the prop. and scan axes:
        prop_pxls_per_scan_step = scan_step_size_um / (
            sample_px_um * np.cos(tilt))
        prop_px_shift_max = int(np.rint(
            prop_pxls_per_scan_step * (scan_steps - 1)))
        scan_steps_per_prop_px = 1 / prop_pxls_per_scan_step
        scan_px_shift_max = int(np.rint(
            scan_steps_per_prop_px * (prop_pxls - 1)))

        # Make image with all projections and flip for traditional view:
        x_pxls = width_pxls
        y_pxls = int(round(
            (prop_pxls + prop_px_shift_max) * np.cos(tilt)))
        z_pxls = int(round(prop_pxls * np.sin(tilt)))
        ln_pxls = separation_line_px_width
        shape = (y_pxls + z_pxls + 2*ln_pxls, x_pxls + z_pxls + 2*ln_pxls)
        return shape
        
    def three_traditional_projections(
        self,
        data,
        scan_step_size_um,
        out=None,
        separation_line_px_width=10,
        ):
        # TODO: consider allowing -ve scan for bi-directional scanning
        # Light-sheet scan, propagation and width axes:
        scan_steps, prop_pxls, width_pxls = data.shape

        # Calculate max pixel shift for shearing on the prop. and scan axes:
        prop_pxls_per_scan_step = scan_step_size_um / (
            sample_px_um * np.cos(tilt))
        prop_px_shift_max = int(np.rint(
            prop_pxls_per_scan_step * (scan_steps - 1)))
        scan_steps_per_prop_px = 1 / prop_pxls_per_scan_step
        scan_px_shift_max = int(np.rint(
            scan_steps_per_prop_px * (prop_pxls - 1)))

        # Make projections:
        O1_proj = np.zeros((prop_pxls + prop_px_shift_max, width_pxls),
                           'uint16')
        scan_proj = np.zeros((prop_pxls, width_pxls), 'uint16')
        width_proj = np.zeros((scan_steps, prop_pxls), 'uint16')

        for i in range(scan_steps):
            prop_px_shift = int(np.rint(i * prop_pxls_per_scan_step))
            target = O1_proj[prop_px_shift:prop_pxls + prop_px_shift, :]
            np.maximum(target, data[i, :, :], out=target) # O1 
            np.maximum(scan_proj, data[i, :, :], out=scan_proj) # scan 
            np.amax(data[i, :, :], axis=1, out=width_proj[i, :]) # width

        # Unshear the width projection:
        unsheared_width_proj = np.zeros(
            (scan_steps + scan_px_shift_max, prop_pxls), 'uint16')
        for i in range(prop_pxls):
            scan_px_shift = int(np.rint(i * scan_steps_per_prop_px))
            unsheared_width_proj[
                scan_px_shift:scan_steps + scan_px_shift, i] = width_proj[:, i]

        # Scale images according to pixel size (divide by X_px_um):
        X_px_um = sample_px_um # width axis
        Y_px_um = sample_px_um * np.cos(tilt) # prop. axis to scan axis
        Z_px_um = sample_px_um * np.sin(tilt) # prop. axis to O1 axis
        O1_img = zoom(O1_proj,(Y_px_um / X_px_um, 1))
        scan_img = zoom(scan_proj,(Z_px_um / X_px_um, 1))
        scan_scale = O1_img.shape[0] / unsheared_width_proj.shape[0]
        # = scan_step_size_um / X_px_um rounded to match O1_img.shape[0]
        width_img = zoom(unsheared_width_proj,(scan_scale, Z_px_um / X_px_um))

        # Make image with all projections and flip for traditional view:
        y_pxls, x_pxls = O1_img.shape
        ln_px = separation_line_px_width # to keep lines of code short!
        ln_min, ln_max = O1_img.min(), O1_img.max()

        if out is None:
            out = np.zeros((y_pxls + scan_img.shape[0]  + 2*ln_px,
                            x_pxls + width_img.shape[1] + 2*ln_px), 'uint16')
            return_value = out
        else:
            return_value = None
        assert out.shape == (y_pxls + scan_img.shape[0]  + 2*ln_px,
                             x_pxls + width_img.shape[1] + 2*ln_px)
        out[ln_px:y_pxls + ln_px, ln_px:x_pxls + ln_px] = O1_img
        out[y_pxls + 2*ln_px:, ln_px:x_pxls + ln_px] = np.flipud(scan_img)
        out[ln_px:y_pxls + ln_px, x_pxls + 2*ln_px:] = np.fliplr(width_img)
        out[y_pxls + 2*ln_px:, x_pxls + 2*ln_px:] = np.full(
            (scan_img.shape[0], width_img.shape[1]), 0)

        # Add line separations between projections:
        out[:ln_px,    :] = ln_max
        out[:ln_px, ::10] = ln_min
        out[y_pxls + ln_px:y_pxls + 2*ln_px,    :] = ln_max
        out[y_pxls + ln_px:y_pxls + 2*ln_px, ::10] = ln_min

        out[:,    :ln_px] = ln_max
        out[::10, :ln_px] = ln_min
        out[:,    x_pxls + ln_px:x_pxls + 2*ln_px] = ln_max
        out[::10, x_pxls + ln_px:x_pxls + 2*ln_px] = ln_min

        out[:] = np.flipud(out)
        return return_value

    def native_view(self, data, scan_step_size_um):
        # Light-sheet scan, propagation and width axes:
        scan_steps, prop_pxls, width_pxls = data.shape
        
        prop_axis_step_size_um = scan_step_size_um * np.cos(tilt)
        native_px_shift = prop_axis_step_size_um / sample_px_um # pick integer
        native_px_shift_max = int(np.rint(native_px_shift * (scan_steps - 1)))
        native_vol = np.zeros(
            (scan_steps, prop_pxls + native_px_shift_max, width_pxls), 'uint16')
        
        for i in range(scan_steps):
            prop_px_shift = int(np.rint(i * native_px_shift))
            native_vol[i, prop_px_shift:prop_pxls + prop_px_shift, :
                       ] = data[i,:,:]
        orthogonal_axis_step_size_um = scan_step_size_um * np.sin(tilt)
        Z_pxls = orthogonal_axis_step_size_um / sample_px_um
        return native_vol, Z_pxls

    def traditional_view(self, native_vol, Z_pxls): # slow but pleasing!
        native_vol_cubic_voxels = zoom(native_vol,(Z_pxls, 1, 1))
        traditional_view = rotate(native_vol_cubic_voxels, np.rad2deg(tilt))
        return traditional_view

if __name__ == '__main__':
    # Set variables: tzcyx acquisition
    # Illumination and emission:
    ch_per_slice = ("LED", "488") # pick any number and order
    pwr_per_ch = (50, 10) # set power for each channel
    fw_pos = 3 # pick 1 filter wheel position per buffer/ao play:
    # 0:blocked, 1:open, 2:ET450/50M, 3:ET525/50M, 4:ET600/50M, 5:ET690/50M
    # 6:ZETquadM

    # Scan settings:
## TODO: figure out how to remove aspect_ratio and deliver slices per volume
## from scan_range_um
    aspect_ratio = 8 # 2 about right for Nyquist?
    shear_px_per_scan_step = 10
    scan_range_um = 25
    
    # Camera chip cropping and exposure time:
    crop_pix_lr = 500
    crop_pix_ud = 500 # max 1019
    ill_time_us = 100 # global exposure

    # Acquisition:
    vol_per_buffer = 1
    num_data_buffers = 3 # increase for multiprocessing
    num_snap = 2 # interbuffer time limited by ao play
    delay_s = 0 # insert delay for time series
    t_stamp = "off"
    max_spiral_tiles = 49

    # Calculate bytes_per_buffer for precise memory allocation:
    roi = pco.legalize_roi({'left': 1 + crop_pix_lr,
                            'right': 2060 - crop_pix_lr,
                            'top': 1 + crop_pix_ud,
                            'bottom': 2048 - crop_pix_ud},
                           camera_type='edge 4.2', verbose=False)
    w_px = roi['right'] - roi['left'] + 1
    h_px = roi['bottom'] - roi['top'] + 1
    slices_per_vol, scan_step_size_um = legalize_scan(
        aspect_ratio, scan_range_um)
    
    legal_aspect_ratio = legalize_voxel_aspect_ratio(aspect_ratio)
    images_per_buffer = vol_per_buffer * slices_per_vol * len(ch_per_slice)
    bytes_per_data_buffer = images_per_buffer * h_px * w_px * 2

    projection_shape = Preprocessor.three_traditional_projections_shape(
        slices_per_vol, h_px, w_px, scan_step_size_um)
    
    bytes_per_preview_buffer = vol_per_buffer * len(ch_per_slice) * int(
        np.prod(projection_shape)) * 2 * max_spiral_tiles

    # Create scope object:
    scope = Snoutscope(
        bytes_per_data_buffer, num_data_buffers, bytes_per_preview_buffer)
    focus_piezo_position_um = (scope.focus_piezo.get_real_position())
    XY_stage_position_mm = scope.XY_stage.get_position()
    
    scope.apply_settings( # Mandatory call
        roi=roi,
        shear_px_per_scan_step=shear_px_per_scan_step,
        illumination_time_microseconds=ill_time_us,
        volumes_per_buffer=vol_per_buffer,
        slices_per_volume=slices_per_vol,
        channels_per_slice=ch_per_slice,
        power_per_channel=pwr_per_ch,
        filter_wheel_position=fw_pos,
        focus_piezo_position_um=focus_piezo_position_um,
        XY_stage_position_mm=XY_stage_position_mm,
        timestamp_mode=t_stamp,
        ).join()

    buffer_time = scope.ao.p2s(scope.voltages.shape[0])

    # Optionally, show voltages. Useful for debugging.
##    scope._plot_voltages()

    # Start frames-per-second timer: acquire, display and save
##    for i in range(num_snap):
##        start = time.perf_counter()
##        scope.snoutfocus(filename='snoutfocus_series%i.tif'%i).join()
##        print('snoutfocus time (s):', (time.perf_counter() - start))
##        scope.snap(
##            display=True,
##            filename='test_images\%06i.tif'%i, # comment out to avoid,
##            delay_seconds=delay_s
##            )
##        t_stamp = "binary+ASCII"
##        scope.apply_settings(timestamp_mode=t_stamp)

    scope.spiral_tiling_preview(num_spirals=1, dx_mm=0.01, dy_mm=0.01)

    scope.close()
