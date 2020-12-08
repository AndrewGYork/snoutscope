#!/usr/bin/python3
# Printing from a child process is tricky:
import io
from contextlib import redirect_stdout
# Showing exceptions from a child process is tricky:
import traceback
# Making sure a child process closes when the parent exits is tricky:
import atexit
# Napari/Qt stuff
import napari
from qtpy.QtCore import QTimer
# Our stuff
from proxy_objects import (
    ProxyManager, _dummy_function, _reconnect_shared_arrays, _SharedNumpyArray)

def display(proxy_manager=None):
    if proxy_manager is None:
        proxy_manager = ProxyManager()
    display = proxy_manager.proxy_object(_NapariDisplay,
                                         custom_loop=_napari_child_loop,
                                         close_method_name='close')
    return display

class _NapariDisplay:
    def __init__(self):
        self.viewer = napari.Viewer()

    def show_image(self, im):
        if not hasattr(self, 'image'):
            self.image = self.viewer.add_image(im)
        else:
            self.image.data = im

    def close(self):
        self.viewer.close()

def _napari_child_loop(child_pipe, shared_arrays,
                       initializer, initargs, initkwargs,
                       close_method_name, closeargs, closekwargs):
    """The event loop of a ProxyObject's child process
    """
    initargs, initkwargs = _reconnect_shared_arrays(
        initargs, initkwargs, shared_arrays)
    closeargs = tuple() if closeargs is None else closeargs
    closekwargs = dict() if closekwargs is None else closekwargs
    state = { # Mutable obj to store variables about the state of the child obj.
        'keep_launching_napari': True,
        'initial_init': True
    }
    while state['keep_launching_napari']:
        with napari.gui_qt():
            # Initialize a napari-specific object
            #
            # If any of the input arguments are _SharedNumpyArrays, we have to
            # show them where to find shared memory:

            # Initialization.
            printed_output = io.StringIO()
            try: # Create an instance of our object...
                with redirect_stdout(printed_output):
                    obj = initializer(*initargs, **initkwargs)
                    # TODO default to "close"
                    if close_method_name is not None:
                        close_method = getattr(obj, close_method_name)
                if state['initial_init']:
                    child_pipe.send(('Successfully initialized',
                                     printed_output.getvalue()))
                    state['initial_init'] = False
            except Exception as e: # If we fail to initialize, just give up.
                ## What if this isnt the initial init?
                e.child_traceback_string = traceback.format_exc()
                child_pipe.send((e, printed_output.getvalue()))
                return None

            command_timer = QTimer()
            def event_loop():
                # print('Looped')
                printed_output = io.StringIO()
                if not child_pipe.poll(): # TODO: check for broken connection?
                    return
                cmd = child_pipe.recv()
                if cmd is None: # This is how the parent signals us to exit.
                    if close_method_name is not None:
                        close_method(*closeargs, **closekwargs)
                    print('Napari close method called')
                    state['keep_launching_napari'] = False
                    return

                attr_name, args, kwargs = cmd
                args, kwargs = _reconnect_shared_arrays(args, kwargs, shared_arrays)
                try:
                    with redirect_stdout(printed_output):
                        result = getattr(obj, attr_name)(*args, **kwargs)
                    if callable(result):
                        result = _dummy_function # Cheaper than a real callable
                    if isinstance(result, _SharedNumpyArray):
                        result = result._disconnect()
                    child_pipe.send((result, printed_output.getvalue()))
                except Exception as e:
                    e.child_traceback_string = traceback.format_exc()
                    child_pipe.send((e, printed_output.getvalue()))

            command_timer.timeout.connect(event_loop)
            command_timer_interval_ms = 1
            command_timer.start(command_timer_interval_ms)
        if state['keep_launching_napari']:
            child_pipe.poll(timeout=None)



# A mocked "microscope" object for testing and demonstrating proxied
# Napari displays. Since this is just for testing our imports are ugly:
class _Microscope:
    def __init__(self):
        import queue
        import time
        self.pm = ProxyManager(shared_memory_sizes=(1*2048*2060*2,))
        self.data_buffers = [
            self.pm.shared_numpy_array(which_mp_array=0,
                                       shape=(1, 2048, 2060),
                                       dtype='uint16')
            for i in range(2)]
        self.data_buffer_queue = queue.Queue()
        for i in range(len(self.data_buffers)):
            self.data_buffer_queue.put(i)
        print("Displaying", self.data_buffers[0].shape,
              self.data_buffers[0].dtype, 'images.')
        self.camera = self.pm.proxy_object(_Camera)
        self.display = display(proxy_manager=self.pm)
        self.num_frames = 0
        self.initial_time = time.perf_counter()

    def snap(self):
        import time
        from proxy_objects import launch_custody_thread
        def snap_task(custody):
            custody.switch_from(None, to=self.camera)
            which_buffer = self.data_buffer_queue.get()
            data_buffer = self.data_buffers[which_buffer]
            self.camera.record(out=data_buffer)
            custody.switch_from(self.camera, to=self.display)
            self.display.show_image(data_buffer)
            custody.switch_from(self.display, to=None)
            self.data_buffer_queue.put(which_buffer)
            self.num_frames += 1
            if self.num_frames == 100:
                time_elapsed =  time.perf_counter() - self.initial_time
                print("%0.2f average FPS"%(self.num_frames / time_elapsed))
                self.num_frames = 0
                self.initial_time = time.perf_counter()
        th = launch_custody_thread(snap_task, first_resource=self.camera)
        return th

class _Camera:
    def record(self, out):
        import numpy as np
        out[:] = np.random.randint(
            0, 2**16, size=out.shape, dtype='uint16')

if __name__ == '__main__':
    scope = _Microscope()
    snap_threads = []
    import threading
    print("Launching a ton of 'snap' threads...")
    for i in range(50):
        try:
            th = scope.snap()
            snap_threads.append(th)
        except RuntimeError:
            print('running threads: ', threading.active_count())
            break
    print(len(snap_threads), "'snap' threads launched.")
    for th in snap_threads:
        th.join()
    print("All 'snap' threads finished execution.")
    input('Hit enter to close napari')
    scope.display.close()
