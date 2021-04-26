# Imports from the python standard library:
import queue
import time
import atexit
import os
from datetime import datetime

# Third party imports, installable via pip:
import numpy as np
from scipy.ndimage import zoom, rotate, gaussian_filter1d
from tifffile import imread, imwrite
# import matplotlib
# We only import matplotlib if/when we call Snoutscope._plot_voltages()

# Our stuff, from github.com/AndrewGYork/tools. Don't pip install.
# One .py file per module, copy files to your local directory.
try:
    # These are only needed if you're making an instance of Snoutscope:
    import pco # Install PCO's SDK to get relevant DLLs
    import ni # Install NI-DAQmx to get relevant DLLs
    import sutter
    import physik_instrumente as p_i
    import thorlabs
    import concurrency_tools as ct
    from napari_in_subprocess import display
except Exception as e:
    import traceback
    print("Import failed:")
    print(traceback.format_exc())
    print("You won't be able to create a Snoutscope object.")

# Snoutscope optical configuration (edit as needed):
M1 = 200 / 2; Mscan = 70 / 70; M2 = 5 / 357; M3 = 200 / 5
MRR = M1 * Mscan * M2; Mtot = MRR * M3;
camera_px_um = 6.5; sample_px_um = camera_px_um / Mtot
tilt = np.deg2rad(30)

class Snoutscope:
    def __init__(
        self,
        num_data_buffers=4 # Memory vs. speed tradeoff (potentially complicated)
        max_bytes_per_data_buffer=2**31 # Bigger is ok, but breaks the TIF spec
        ):
        self.unfinished_tasks = queue.Queue()
        self.available_data_buffer_names = queue.Queue()
        self.data_buffers = {}
        # Initialize many slow things in parallel via threads:
        init_fw = ct.ResultThread(target=self._init_filter_wheel).start()
        init_camera = ct.ResultThread(target=self._init_camera).start()
        init_snoutfocus = ct.ResultThread(target=self._init_snoutfocus).start()
        init_focus = ct.ResultThread(target=self._init_focus_piezo).start()
        init_stage = ct.ResultThread(target=self._init_XY_stage).start()
        # ...and initialize a few less-slow things in the main thread:
        self._init_display()         # ~1.3 s
        self._init_processor()       # ~0.8 s
        self._init_ao()              # ~0.2 s
        # Check our threads for exceptions, from fastest to slowest:
        init_stage.get_result()      # ~0.4 s
        init_focus.get_result()      # ~0.6 s
        init_snoutfocus.get_result() # ~1.0 s
        init_camera.get_result()     # ~3.6 s
        init_fw.get_result()         # ~5.3 s
        print('Finished initializing Snoutscope hardware')
        print("Don't forget, you must now call .apply_settings() with",
              "all arguments specified before you can call .acquire()")

    def apply_settings(
        self,
        # Camera settings:
        illumination_time_microseconds=None, # Float
        timestamp_mode=None, # String, see pco.py ._set_timestamp_mode()
        roi=None,            # Dict,   see pco.py ._set_roi()
        # Color channel settings:
        channels_per_slice=None,    # Tuple of strings
        power_per_channel=None,     # Tuple of floats
        filter_wheel_position=None, # Int
        # Galvo scan settings:
        scan_step_size_px=None,  # Int, or Float but be careful with Float!
        slices_per_volume=None,  # Int
        volumes_per_buffer=None, # Int
        # Field-of-view settings:
        focus_piezo_position_um=None, # Float or (Float, "relative")
        XY_stage_position_mm=None,    # (Float, Float, optional: "relative")
        ):
        '''Rapidly change multiple hardware settings concurrently.

        .apply_settings() launches a 'custody thread' - a type of thread
        that knows how to politely share resources with other custody
        threads launched by methods like .acquire().

        .apply_settings() returns this thread immediately, but the
        thread may take a while to finish. While tasks are active, don't
        change attributes of the Snoutscope object.

        To "regain custody" of the Snoutscope object, call .finish_all_tasks()
        '''
        settings = locals()
        settings.pop('self') # 'self' is not a setting...
        def settings_task(custody):
            # If another thread is using the camera, this thread
            # shouldn't change hardware settings (e.g. the filter wheel).
            # Once this thread gets custody of the camera, it's ok to
            # change most hardware settings:
            custody.switch_from(None, to=self.camera)
            self._settings_are_sane = False # In case this thread crashes
            for setting, value in settings.items(): 
                if value is None:
                    # If a setting was already set by a previous call to
                    # .apply_settings(), and the current requested value
                    # is "None" (the default value for all arguments to
                    # .apply_settings()), then we leave the setting
                    # unchanged. This can save a lot of time.
                    pass 
                else:
                    setattr(self, setting, value) # A lot like self.x = x
                # At this point, all settings must be specified:
                assert hasattr(self, setting), (
                    "Attribute '%s' must be set by .apply_settings()"%k)
            # Send hardware commands, slowest to fastest:
            if XY_stage_position_mm is not None:
                x, y = XY_stage_position_mm[0:2]
                if len(XY_stage_position_mm) > 2: # Relative motion?
                    assert XY_stage_position_mm[2] == "relative"
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
                if roi is not None:
                    self.roi = self.camera._set_roi(roi)
                    self.ud_px = self.roi['bottom'] - self.roi['top'] + 1
                    self.lr_px = self.roi['right'] - self.roi['left'] + 1
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
                self.focus_piezo_position_um = ( 
                   self.focus_piezo.get_real_position())
            if filter_wheel_position is not None:
                self.filter_wheel._finish_moving()
            if XY_stage_position_mm is not None:
                self.XY_stage.finish_moving()
                self.XY_stage_position_mm = self.XY_stage.get_position()
            # Check data buffer size:
            bytes_per_data_buffer = (self.volumes_per_buffer *
                                     self.slices_per_volume *
                                     len(self.channels_per_slice) *
                                     self.ud_px* self.lr_px * 2)
            if bytes_per_data_buffer > self.max_bytes_per_data_buffer:
                raise MemoryError((
                    "Snoutscope settings need a data buffer with %d bytes" +
                    "but max_bytes_per_data_buffer=%d")%(
                        bytes_per_data_buffer, self.max_bytes_per_data_buffer))
            self._settings_are_sane = True
            custody.switch_from(self.camera, to=None) # Release camera
        settings_thread = ct.CustodyThread(
            target=settings_task, first_resource=self.camera).start()
        self.unfinished_tasks.put(settings_thread)
        return settings_thread

    def acquire(
        self,
        delay_seconds=None,
        display=True,
        data_filename=None,
        projection_filename=None,
        preview_filename=None,
        ):
        '''Rapidly acquire one file worth of images synchronized via one
        continuous play of the analog-out card.

        .acquire() launches a 'custody thread' - a type of thread that
        knows how to politely share resources with other custody threads
        launched by methods like .apply_settings().

        .acquire() returns this thread immediately, but the thread may
        take a while to finish. While tasks are active, don't change
        attributes of the Snoutscope object.

        To "regain custody" of the Snoutscope object, call .finish_all_tasks()
        '''
        if delay_seconds is not None and delay_seconds > 3:
            # We're going to wait long enough before this acquisition
            # that we might as well run a hardware drift correction:
            self.snoutfocus(delay_seconds=delay_seconds)
            delay_seconds = None
        def acquisition_task(custody):
            # If another thread is using the camera, this thread
            # should wait in line until the camera is available:
            custody.switch_from(None, to=self.camera)
            start_time = time.perf_counter()
            assert hasattr(self, '_settings_are_sane'), (
                'Call .apply_settings() before using .acquire()')
            assert self._settings_are_sane, (
                '.apply_settings() failed. Call it again before .acquire().')
            # We might as well start writing voltages to the AO card now:
            write_voltages_thread = ct.ResultThread(
                target=self.ao._write_voltages, args=(self.voltages,)).start()
            # We might as well allocate memory now too:
            num_exposures = (len(self.channels_per_slice) *
                             self.slices_per_volume *
                             self.volumes_per_buffer)
            data_buffer = self._get_data_buffer(
                (num_exposures, self.ud_px, self.lr_px), 'uint16')
            write_voltages_thread.get_result()
            # If we want to pause between acquisitions:
            if delay_seconds is not None:
                sleep(delay_seconds, start_time) # Hopefully fairly precise
            # camera.record_to_memory() blocks, so we use a thread:
            camera_thread = ct.ResultThread(
                target=self.camera.record_to_memory,
                kwargs={'num_images': exposures_per_buffer,
                        'out': data_buffer,
                        'first_trigger_timeout_seconds': 1},).start()
            # There's a race here. The PCO camera starts with N empty
            # single-frame buffers (typically 16), which are filled by
            # the triggers sent by ao.play_voltages(). The camera_thread
            # empties them, hopefully fast enough that we never run out.
            # So far, the camera_thread seems to both start on time, and
            # keep up reliably once it starts, but this could be
            # fragile. The camera thread (effectively) acquires shared
            # memory as it writes to the allocated buffer. On this
            # machine the memory acquisition is faster than the camera
            # (~4 GB/s vs ~1 GB/s) but this could also be fragile if
            # another process interferes.
            self.ao.play_voltages(block=False)
            camera_thread.get_result()
            # Acquisition is 3D, but display and filesaving are 5D:
            data_buffer = data_buffer.reshape(self.volumes_per_buffer,
                                              self.slices_per_volume,
                                              len(self.channels_per_slice),
                                              data_buffer.shape[-2],
                                              data_buffer.shape[-1])
            # Do we need to compute projections?
            if (display, projection_filename, preview_filename) == (
                False, None, None):
                custody.switch_from(self.camera, to=None) # Nope
            else:
                # Compute traditional XYZ projections of the data buffer
                custody.switch_from(self.camera, to=self.processor)
                if self.timestamp_mode != 'off':
                    # Temporarily zero out timestamp rows so they don't
                    # trash our projections:
                    timestamps = np.copy(data_buffer[:, :, :, 0:8, :])
                    data_buffer[:, :, :, 0:8, :] = 0
                # Set some temporary short nicknames just for brevity:
                vpb_cps = (self.volumes_per_buffer,
                           len(self.channels_per_slice))
                p = self.processor
                # Allocate shared memory to hold our projections.
                p.set_data_shape((self.slices_per_volume, self.ud_px, self.lr_px),
                                 self.scan_step_size_px)
                xy_proj = ct.SharedNDArray(vpb_cps + (p.y_px, p.x_px), 'uint16')
                xz_proj = ct.SharedNDArray(vpb_cps + (p.z_px, p.x_px), 'uint16')
                yz_proj = ct.SharedNDArray(vpb_cps + (p.z_px, p.y_px), 'uint16')
                preview = ct.SharedNDArray(vpb_cps + p.preview_shape, 'uint16')
                for vo in range(data_buffer.shape[0]):
                    for ch in range(data_buffer.shape[2]):
                        self.processor.set_data(data_buffer[vo, :, ch, :, :])
                        self.processor.calculate_traditional_projections(
                            xy_out=xy[vo, ch, :, :],
                            xz_out=xz[vo, ch, :, :],
                            yz_out=yz[vo, ch, :, :],
                            preview_out=preview[vo, ch, :, :])
                self.processor.clear_data()
                if self.timestamp_mode != 'off':
                    # Restore our timestamp rows
                    data_buffer[:, :, :, 0:8, :] = timestamps
                # Optionally show the live preview we just calculated:
                if display:
                    custody.switch_from(self.processor, to=self.display)
                    self.display.show_image(preview)
                    custody.switch_from(self.display, to=None)
                else:
                    custody.switch_from(self.processor, to=None)
            # Optionally save files to disk
            if data_filename is not None:
                print("Saving file:", data_filename)
                imwrite(data_filename, data_buffer, imagej=True)
            self._release_data_buffer(data_buffer)
            if projection_filename is not None:
                for data, suffix in ((xy, "xy"), (xz, "xz"), (yz, "yz")):
                    root, ext = os.path.splitext(projection_filename)
                    fn = root + "_" + suffix + ext
                    imwrite(fn, data, imagej=True)
            if preview_filename is not None:
                print("Saving file:", preview_filename)
                imwrite(preview_filename, preview, imagej=True)
            return None
        acquisition_thread = ct.CustodyThread(
            target=acquisition_task, first_resource=self.camera).start()
        self.unfinished_tasks.put(acquisition_thread)
        return acquisition_thread

# TODO: How shall this interact with .acquire()?
##    def _save_metadata(self, filename, delay_seconds, path):
##        to_save = {
##            'Date':datetime.strftime(datetime.now(),'%Y-%m-%d'),
##            'Time':datetime.strftime(datetime.now(),'%H:%M:%S'),
##            'filename':filename,
##            'delay_seconds':delay_seconds,
##            'channels_per_slice':self.channels_per_slice,
##            'power_per_channel':self.power_per_channel,
##            'filter_wheel_position':self.filter_wheel_position,
##            'illumination_time_us':self.illumination_time_microseconds,
##            'timestamp_mode':self.timestamp_mode,
##            'roi':self.roi,
##            'scan_step_size_px':self.scan_step_size_px,
##            'scan_step_size_um':calculate_scan_step_size_um(
##                self.scan_step_size_px),
##            'slices_per_volume':self.slices_per_volume,
##            'scan_range_um':calculate_scan_range_um(
##                self.scan_step_size_px, self.slices_per_volume),
##            'volumes_per_buffer':self.volumes_per_buffer,
##            'focus_piezo_position_um':self.focus_piezo_position_um,
##            'XY_stage_position_mm':self.XY_stage_position_mm,
##            'preview_line_px_width':self.preview_line_px_width,
##            'MRR':MRR,
##            'Mtot':Mtot,
##            'tilt':tilt,
##            'sample_px_um':sample_px_um,
##            'voxel_aspect_ratio':calculate_voxel_aspect_ratio(
##                self.scan_step_size_px),
##            }
##        with open(os.path.splitext(path)[0] + '.txt', 'w') as file:
##            for k, v in to_save.items():
##                file.write(k + ': ' + str(v) + '\n')

    def snoutfocus(self, filename=None, delay_seconds=None):
        def snoutfocus_task(custody):
            custody.switch_from(None, to=self.camera) # Safe to change settings
            if delay_seconds is not None:
                start_time = time.perf_counter()
                if delay_seconds > 3: # 3 seconds is def. enough time to focus
                    time.sleep(delay_seconds - 3)
            assert self._settings_are_sane, (
                'Did .apply_settings() fail? Please call it again, ' +
                'with all arguments specified.')
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
            camera_thread = ct.ResultThread(
                target=self.camera.record_to_memory,
                args=(exposures_per_buffer,),
                kwargs={'out': data_buffer,
                        'first_trigger_timeout_seconds': 1},).start()
            self.ao.play_voltages(voltages, block=False) # Ends at 0 V
            camera_thread.get_result()
            # Start cleaning up after ourselves:
            self.filter_wheel.move(old_filter_wheel_position,
                                   speed=6, block=False)
            # Inspect the images to find/set best snoutfocus piezo position:
            # Top rows might have timestamps
            if np.max(data_buffer[:,8:,:]) < 5 * np.min(data_buffer[:,8:,:]):
                print('WARNING: snoutfocus laser intensity is low:',
                      'is the laser/shutter powered up?')
            controller_voltage = piezo_voltage_step * np.unravel_index(
                np.argmax(data_buffer[:,8:,:]), data_buffer[:,8:,:].shape)[0]
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
                sleep(delay_seconds, start_time)
            custody.switch_from(self.camera, to=None)
            if filename is not None:
                print("Saving file:", filename)
                imwrite(filename, data_buffer[:, np.newaxis, :, :], imagej=True)
            self._release_data_buffer(data_buffer)
        snoutfocus_thread = ct.CustodyThread(
            target=snoutfocus_task, first_resource=self.camera).start()
        self.unfinished_tasks.put(snoutfocus_thread)
        return snoutfocus_thread

##    def spiral_tiling_preview(
##        self,
##        num_spirals=1,
##        dx_mm=0.1,
##        dy_mm=0.1,
##        ):
##        ####################
##        # WORK IN PROGRESS #
##        ####################
##        # TODO: xy, or yx? What's stage leftright vs. screen leftright?
##        # If we get this straight, this code will suck less.
#### TODO: - make sure stage is at max velocity and then return to previous
####       - Find some way off allowing users to pick + move to a field of choice
##        assert dx_mm and dy_mm < 1
##        assert num_spirals < 4 # C'mon don't be silly, this is still a lot
##        data_filename = "spiral_%02i_%02i.tif"
##        preview_filename = "spiral_%02i_%02i_preview.tif"
##        ix, iy = [0], [0] # Mutable, integer coords relative to starting tile
##        acquire_tasks = []
##        def move(x, y):
##            ix[0], iy[0] = ix[0]+x, iy[0]+y
##            self.apply_settings(
##                XY_stage_position_mm=(dx_mm*x, dy_mm*y, "relative"))
##            acquire_tasks.append(self.acquire(
##                filename=data_filename%(ix[0]+num_spirals, iy[0]+num_spirals),
##                display=True))
##        def move_up():    move( 0,  1)
##        def move_down():  move( 0, -1)
##        def move_left():  move(-1,  0)
##        def move_right(): move( 1,  0)
##        move(0, 0) # get the central tile
##        for which_spiral in range(num_spirals): # This took me a while...
##            for i in range(1):
##                move_up()
##            for i in range(1 + 2*which_spiral):
##                move_right()
##            for i in range(2 + 2*which_spiral):
##                move_down()
##            for i in range(2 + 2*which_spiral):
##                move_left()
##            for i in range(2 + 2*which_spiral):
##                move_up()
##        self.apply_settings( # Return to original position
##            XY_stage_position_mm=(-dx_mm*ix[0], -dy_mm*iy[0], "relative"))
##        def display_preview_task(custody):
##            # To preserve order we follow the same custody pattern as acquire():
##            custody.switch_from(None, to=self.camera)
##            custody.switch_from(self.camera, to=self.preprocessor)
##            custody.switch_from(self.preprocessor, to=self.display)
##            # You can't display until the preview files have saved to disk:
##            for a in acquire_tasks: a.join()
##            # Now you can start to reload the preview images:
##            # TODO: re-try (with small delay) if file not found yet?
##            first_tile = imread(preview_filename%(0, 0))
##            ud_pix, lr_pix = first_tile.shape[-2:]
##            preview_shape = list(first_tile.shape)
##            preview_shape[-2] = ud_pix * (1 + 2*num_spirals)
##            preview_shape[-1] = lr_pix * (1 + 2*num_spirals)
##            # We're ready to hog some shared resources:
##            # TODO: seriously consider adding spiral_preview_buffers to avoid
##            # asking for memory that's not available
##            preview_buffer = self._get_preview_buffer(preview_shape, 'uint16')
##            preview_buffer[..., :ud_pix, :lr_pix] = first_tile
##            for i in range(1 + 2*num_spirals):
##                for j in range(1 + 2*num_spirals):
##                    if (i, j) == (0, 0): continue # We already did this tile
##                    preview_buffer[...,
##                                   ud_pix*j:ud_pix*(j+1),
##                                   lr_pix*i:lr_pix*(i+1)
##                                   ] = imread(preview_filename%(i, j))
##            self.display.show_image(preview_buffer)
##            custody.switch_from(self.display, to=None)
##            self._release_preview_buffer(preview_buffer)
##        display_thread = proxy_objects.launch_custody_thread(
##            target=display_preview_task, first_resource=self.camera)
##        self.unfinished_tasks.put(display_thread)

    def finish_all_tasks(self):
        collected_tasks = []
        while True:
            try:
                th = self.unfinished_tasks.get_nowait()
            except queue.Empty:
                break
            th.get_result()
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
        self.camera = ct.ObjectInSubprocess(pco.Camera, verbose=False,
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

    def _init_processor(self):
        print("Initializing processor...")
        self.processor = ct.ObjectInSubprocess(Processor)
        print("done with processor.")

    def _init_display(self):
        print("Initializing display...")
        self.display = display()
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
        # Calculate scan_range_um and check limit:
        scan_range_um = calculate_scan_range_um(
            self.scan_step_size_px, self.slices_per_volume)
        assert 0 <= scan_range_um <= 200 # optical limit
        # Calculate galvo voltages from volume settings:
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
        """Instantiating a large numpy array (with or without shared
        memory) is very fast, but the first time you *write* to the
        array is unusually slow.

        We seem to be able to keep up with our camera even with the
        first (slow) write, but it seems wasteful to discard the memory
        after. Here, we recycle previously-used shared-memory arrays
        that are large enough for current needs. This probably doesn't
        matter, but it's possible it has some impact on reliability.
        """
        # This will run once on init, making tiny placeholders:
        while len(self.data_buffers) < self.num_data_buffers:
            db = ct.SharedNDArray(shape=1, dtype='uint8')
            self.data_buffers[db.shared_memory.name] = db
            self.available_data_buffer_names.put(db.shared_memory.name)
        # Now we can assume all data buffers exist. Wait for a buffer:
        name = self.available_data_buffer_names.get()
        # We've got a buffer; is it big enough?
        requested_bytes = np.prod(shape, dtype='uint64') * dtype.itemsize
        if requested_bytes <= self.data_buffers[name].nbytes:
            # The existing buffer is big enough. Return an
            # appropriately-sized view.
            data_buffer = self.data_buffers[name]
            data_buffer = data_buffer.view('uint8').ravel()[:requested_bytes]
            data_buffer = data_buffer.view(dtype).reshape(shape)
        else:
            # Destroy the old buffer and create a fresh one.
            del self.data_buffers[name]
            data_buffer = ct.SharedNDArray(shape, dtype)
            self.data_buffers[data_buffer.shared_memory.name] = data_buffer
        assert data_buffer.shape, data_buffer.dtype = shape, dtype
        return data_buffer

    def _release_data_buffer(self, x):
        assert isinstance(x, ct.SharedNDArray)
        assert x.shared_memory.name in self.data_buffers
        self.available_data_buffer_names.put(x.shared_memory.name)
        return None

# A convenience function for semiprecise waiting:
def sleep(seconds, start_time=None):
    if start_time is None:
        start_time = time.perf_counter()
    while (time.perf_counter() - start_time) < (seconds - 0.15):
        time.sleep(0.1) # Big imprecise sleeps to eat up big chunks of time
    while (time.perf_counter() - start_time) < seconds:
        pass # Expensive but precise busy-waiting for the last few ms

# Convenience functions for Snoutscope geometry calculations:
def calculate_scan_step_size_um(scan_step_size_px):
    return scan_step_size_px * sample_px_um / np.cos(tilt)

def calculate_scan_range_um(scan_step_size_px, slices_per_volume):
    scan_step_size_um = calculate_scan_step_size_um(scan_step_size_px)
    return scan_step_size_um * (slices_per_volume - 1)

def calculate_voxel_aspect_ratio(scan_step_size_px):
    return scan_step_size_px * np.tan(tilt)

def calculate_cuboid_voxel_scan(voxel_aspect_ratio, scan_range_um):
    scan_step_size_px = max(int(round(voxel_aspect_ratio / np.tan(tilt))), 1)
    scan_step_size_um = calculate_scan_step_size_um(scan_step_size_px)
    slices_per_volume = 1 + int(round(scan_range_um / scan_step_size_um))
    return scan_step_size_px, slices_per_volume # watch out for fencepost!

class Processor:
    """This object calculates useful transformations of raw snoutscope data.
    """
    def __init__(self):
        self.clear_data()
        self.divider_px = 3
        return None

    def set_data(self, data, scan_step_size_px):
        """Sets the raw 3D data from which we'll calculate processed data
        """
        self.set_data_shape(data.shape, scan_step_size_px)
        self.data = data
        return None

    def clear_data(self):
        self.data = None
        self.projections = None
        self.native_view = None
        self.traditional_view = None
        return None

    def set_data_shape(self, shape, scan_step_size_px):
        """Even if the raw data doesn't exist yet, we can calculate many
        useful quantities from the shape of the raw data.
        """
        assert len(shape) == 3
        self.data_shape = shape
        self.scan_step_size_px = scan_step_size_px
        self.clear_data()
        # Calculate scan size in projection coordinates:
        scan_steps, prop_px, width_px = shape
        self.prop_px_shift_max = int(round((scan_steps-1) * scan_step_size_px))
        self.scan_px_shift_max = int(round((prop_px - 1)  / scan_step_size_px))
        # Calculate the pixel dimensions of projections and embeddings:
        self.x_px = width_px
        self.y_px = int(round(np.cos(tilt) * (prop_px + prop_px_shift_max)))
        self.z_px = int(round(np.sin(tilt) * prop_px))
        self.preview_shape = (self.y_px + self.z_px + 2*self.divider_px,
                              self.x_px + self.z_px + 2*self.divider_px)
        return None
                        
    def calculate_traditional_projections(
        self,
        projection_type='maximum', # 'maximum' or 'sum'
        # Optional preallocated arrays to hold output:
        xy_out=None,
        xz_out=None,
        yz_out=None,
        preview_out=None,
        ):
        """Project raw Snoutscope data to familiar orthogonal views.

        The 'xy' view is the view most users expect: the viewing
        direction for this projection is along the optical axis of the
        primary objective. This view is especially useful for
        interactive 'joystick' navigation. For wide, flat samples that
        hug the coverslip, this projection is often much more
        informative than the other two views.

        The 'yz' viewing direction is the same direction that the galvo
        scans. The 'xz' viewing direction is the direction in which the
        light sheet is wide.
        """
        assert self.data is not None
        assert projection_type in ('maximum', 'sum')
        # Check preallocated output buffer shape and dtype:
        for out, shape in ((xy_out, (self.y_px, self.x_px)),
                           (xz_out, (self.z_px, self.x_px)),
                           (yz_out, (self.z_px, self.y_px)),
                           (preview_out, self.preview_shape)):
            if out is not None:
                assert out.shape == shape, 'Wrong output buffer shape'
                assert out.dtype == 'uint16', 'Wrong output buffer dtype'
        # These arrays will hold our projections, but they don't all
        # have the same pixel size yet:
        scan_steps, prop_px, width_px = self.data_shape
        O1_proj =    np.zeros((prop_px + prop_px_shift_max, width_px), 'uint16')
        scan_proj =  np.zeros((prop_px,                     width_px), 'uint16')
        width_proj = np.zeros((scan_steps,                   prop_px), 'uint16')
        # Make projections:
        if projection_type == 'maximum':
            proj_2d, proj_1d = np.maximum, np.amax
        elif projection_type == 'sum':
            proj_2d, proj_1d = np.add, np.sum
        for i in range(scan_steps):
            prop_px_shift = int(round(i * self.scan_step_size_px))
            s = slice(prop_px_shift, prop_px_shift + prop_px)
            proj_2d(O1_proj[s, :], self.data[i, :, :], out=O1_proj[s, :])
            proj_2d(scan_proj,     self.data[i, :, :], out=scan_proj)
            proj_1d(self.data[i, :, :], axis=1,        out=width_proj[i, :])
        # The 'width projection' we just calculated is sheared. Unshear it:
        unsheared_width_proj = np.zeros(
            (scan_steps + self.scan_px_shift_max, prop_pxls), 'uint16')
        for i in range(prop_pxls):
            scan_px_shift = int(round(i / self.scan_step_size_px))
            s = slice(scan_px_shift, scan_px_shift + scan_steps)
            unsheared_width_proj[s, i] = width_proj[:, i]
        # Scale projections so they have the same pixel size as the camera:
        xy_out = zoom(O1_proj,   (np.cos(tilt), 1), output=xy_out)
        xz_out = zoom(scan_proj, (np.sin(tilt), 1), output=xz_out)
        yz_out = zoom(unsheared_width_proj,
                      (self.y_px / unsheared_width_proj.shape[0], np.sin(tilt)),
                      output=yz_out)
        # Make image with all projections and flip for 'traditional' preview:
        if preview_out is None:
            preview_out = np.zeros(self.preview_shape 'uint16')
        im_min, im_max = xy_out.min(), xy_out.max()
        x_px, y_px, ln_px = self.x_px self.y_px, self.divider_px
        preview_out[ln_px:ln_px + y_px, ln_px:ln_px + x_px] = xy_out
        preview_out[y_px + 2*ln_px:,    ln_px:ln_px + x_px] = xz_out[::-1, :]
        preview_out[ln_px:ln_px + y_px, x_px + 2*ln_px:   ] = yz_out[:, ::-1]
        preview_out[y_px + 2*ln_px:,    x_px + 2*ln_px:] = im_min
        # Add line separations between projections:
        preview_out[:ln_px,    :] = im_max
        preview_out[:ln_px, ::10] = im_min
        preview_out[y_px + ln_px:y_px + 2*ln_px,    :] = im_max
        preview_out[y_px + ln_px:y_px + 2*ln_px, ::10] = im_min
        preview_out[:,    :ln_px] = im_max
        preview_out[::10, :ln_px] = im_min
        preview_out[:,    x_px + ln_px:x_px + 2*ln_px] = im_max
        preview_out[::10, x_px + ln_px:x_px + 2*ln_px] = im_min
        preview_out[:] = out[::-1, :]
        self.traditional_projections = {
            'xy': xy_out, 'xz': xz_out, 'yz': yz_out, 'preview': preview_out}
        return self.traditional_projections

    def _set_preview_divider_px(self, pixels):
        """You might want to make this line thicker or thinner.
        """
        assert pixels >= 0
        self.divider_px = int(pixels)
        if hasattr(self, 'preview_shape'):
            self.preview_shape = (self.y_px + self.z_px + 2*self.divider_px,
                                  self.x_px + self.z_px + 2*self.divider_px)
        return None

    def calculate_native_view(self, out=None):
        """The 'native view' is the most principled view of the data for
        quantitative analysis. If 'scan_step_size_px' is an integer,
        then no interpolation is needed to view  the volume. The native
        view looks at the sample with the 'tilt' of Snouty,
        perpendicular to the light sheet.
        """
        assert self.data is not None
        # Light-sheet scan, propagation and width axes:
        scan_steps, prop_px, width_px = self.data_shape
        out_shape = (scan_steps, prop_px + self.scan_px_shift_max, width_px)
        if out is None:
            self.native_view = np.zeros(out_shape, dtype='uint16')
        else:
            assert out.shape == out_shape
            assert out.dtype == 'uint16'
            self.native_view = out
        for i in range(scan_steps):
            prop_px_shift = int(round(i * self.scan_step_size_px))
            s = slice(prop_px_shift, prop_px_shift + prop_px)
            self.native_view[i, s, :] = self.data[i, :, :]
        return self.native_view

    def calculate_traditional_view(self, out=None):
        """Very slow but pleasing - rotates the native view to the
        traditional view!

        I'm sure we could make a faster version of this, but for now
        this is usable.
        """
        assert self.native_view is not None
        native_view_with_cubic_voxels = zoom(
            self.native_view, (self.scan_step_size_px * np.tan(tilt), 1, 1))
        self.traditional_view = rotate(
            native_view_with_cubic_voxels, np.rad2deg(tilt), output=out)
        return self.traditional_view

# TODO: think about how to organize these tools
##class Postprocessor:
##    # 'estimate_sample_z_axis_px' can be used for software autofocus. It uses
##    # an earlier 'preview' image to estimate the z location of the sample in
##    # pixels. Multiply by 'sample_px_um' for an absolute value in z. Choose:
##    # - 'max_intensity' to track the brightest z pixel
##    # - 'max_gradient' as a proxy for the coverslip boundary z pixel
##    def estimate_sample_z_axis_px(
##        self,
##        preview_image, # 2D preview: single volume, single channel
##        roi,
##        timestamp_mode,
##        preview_line_px_width,
##        method='max_gradient',
##        gaussian_filter_std=3,
##        ):
##        assert method in ('max_intensity', 'max_gradient')
##        w_px = roi['right'] - roi['left'] + 1
##        h_px = roi['bottom'] - roi['top'] + 1
##        if timestamp_mode != "off": h_px = h_px - 8 # skip timestamp rows
##        z_px = int(round(h_px * np.sin(tilt))) # Preprocessor definition
##        inspect_me = preview_image[:z_px, preview_line_px_width:w_px]
##        intensity_line = np.average(inspect_me, axis=1)[::-1] # O1 -> coverslip
##        intensity_line_smooth = gaussian_filter1d(
##            intensity_line, gaussian_filter_std) # reject hot pixels 
##        if method == 'max_intensity':
##            return np.argmax(intensity_line_smooth)
##        max_intensity = np.max(intensity_line_smooth)
##        intensity_gradient = np.zeros((len(intensity_line_smooth) - 1))
##        for px in range(len(intensity_line_smooth) - 1):
##            intensity_gradient[px] = (
##                intensity_line_smooth[px + 1] - intensity_line_smooth[px])
##            if intensity_line_smooth[px + 1] == max_intensity:
##                break
##        return np.argmax(intensity_gradient)
##
##    # 'estimate_roi' can be used for cropping empty pixels from raw data.
##    # The snoutscope produces vast amounts of data very quickly, often with
##    # many empty pixels - discarding them can help manage the deluge.
##    def estimate_roi(
##        self,
##        volume, # raw data 3D: single volume, single channel
##        timestamp_mode,
##        signal_to_bg_ratio=1.2, # adjust for threshold
##        gaussian_filter_std=3, # adjust for smoothing/hot pixel rejection
##        ):
##        scan_px, prop_px, width_px = volume.shape
##        ts_px = 0
##        if timestamp_mode != "off": ts_px = 8 # skip timestamp rows
##        # Max project volume to images:
##        width_projection = np.amax(volume[:,ts_px:,:], axis=2)
##        scan_projection  = np.amax(volume[:,ts_px:,:], axis=0)
##        # Max project images to lines and smooth to reject hot pixels:
##        scan_line  = gaussian_filter1d(
##            np.max(width_projection, axis=1), gaussian_filter_std)
##        prop_line  = gaussian_filter1d(
##            np.max(scan_projection, axis=1), gaussian_filter_std)
##        width_line = gaussian_filter1d(
##            np.max(scan_projection, axis=0), gaussian_filter_std)
##        # Find background level and set threshold:
##        scan_threshold  = int(min(scan_line)  * signal_to_bg_ratio)
##        prop_threshold  = int(min(prop_line)  * signal_to_bg_ratio)
##        width_threshold = int(min(width_line) * signal_to_bg_ratio)
##        # Estimate roi:
##        i_min, i_max = [0, 0, 0], [scan_px - 1, prop_px - 1, width_px - 1]
##        for i in range(scan_px):
##            if scan_line[i]  > scan_threshold:
##                i_min[0] = i
##                break
##        for i in range(prop_px):
##            if prop_line[i]  > prop_threshold:
##                i_min[1] = i + ts_px # put timestamp rows back
##                break
##        for i in range(width_px):
##            if width_line[i] > width_threshold:
##                i_min[2] = i
##                break        
##        for i in range(scan_px):
##            if scan_line[-i] > scan_threshold:
##                i_max[0] = i_max[0] - i
##                break
##        for i in range(prop_px):
##            if prop_line[-i] > prop_threshold:
##                i_max[1] = i_max[1] - i - ts_px # put timestamp rows back
##                break
##        for i in range(width_px):
##            if width_line[-i] > width_threshold:
##                i_max[2] = i_max[2] - i
##                break
##        return i_min, i_max


if __name__ == '__main__':
    # TODO: should we record this somewhere like we do with n2c?
    ## filter wheel options:        0:blocked,      1:open
    # 2:ET450/50M,  3:ET525/50M,    4:ET600/50M,    5:ET690/50M
    # 6:ZETquadM,   7:empty         8:empty         9:empty

    # Scan: input user-friendly options -> return values for .apply_settings()
    scan_step_size_px, slices_per_volume = calculate_cuboid_voxel_scan(
        aspect_ratio=2,
        scan_range_um=50,)
    
    # Camera chip cropping:
    crop_px_lr = 500
    crop_px_ud = 900 # max 1019

    # Create scope object:
    scope = Snoutscope(
        num_data_buffers=4,
        max_bytes_per_data_buffer=2**31)
    scope.apply_settings( # Mandatory call
        channels_per_slice=("LED", "488"),
        power_per_channel=(50, 10), # 0-100% match to channels
        filter_wheel_position=3, # pick 1 position - see above options
        illumination_time_microseconds=100,
        timestamp_mode="off", # pick: "off", "binary", "binary+ASCII", "ASCII"
        roi={'left': 1 + crop_px_lr,
             'right': 2060 - crop_px_lr,
             'top': 1 + crop_px_ud,
             'bottom': 2048 - crop_px_ud},
        scan_step_size_px=scan_step_size_px,
        slices_per_volume=slices_per_volume,
        volumes_per_buffer=1,
        focus_piezo_position_um=(0,'relative'),
        XY_stage_position_mm=(0,0,'relative'),
        )

    # Run snoufocus and acquire:
    os.makedirs('test_data', exist_ok=True)
    scope.snoutfocus(filename='test_data\init.tif')
    for i in range(2):
        scope.acquire(
            display=True,
            filename='test_data\%06i.tif'%i, # comment out to avoid
            delay_seconds=0
            )
##    scope.spiral_tiling_preview(num_spirals=1, dx_mm=0.01, dy_mm=0.01)
    scope.finish_all_tasks()
    scope.close()
