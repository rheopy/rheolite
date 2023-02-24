import cv2
import numpy as np
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import lmfit
import pandas as pd


class CaberVideo:
    """Container for capillary breakup videos.
    
    Attributes:
        video_filename (str): The filename of the video file.
        cap (cv2.VideoCapture): The OpenCV video capture object.
        num_frames (int): The total number of frames in the video.
        mask (tuple): The mask to be applied to each frame of the video.
    """
    
    def __init__(self, video_filename, mask=(slice(None), slice(None))):
        """Initializes a CaberVideo object.
        
        Args:
            video_filename (str): The filename of the video file.
            mask (tuple): The mask to be applied to each frame of the video. Defaults to (slice(None), slice(None)).
        """
        self.video_filename = video_filename
        self.cap = cv2.VideoCapture(self.video_filename)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._mask = mask
    
    def __iter__(self):
        """Returns an iterator for the CaberVideo object."""
        return self
    
    def __next__(self):
        """Returns the next frame in the video."""
        ret, frame = self.cap.read()
        if ret:
            return frame[self.mask]
        else:
            self.cap.release()
            raise StopIteration
    
    def __getitem__(self, index):
        """Returns the frame at the given index in the video."""
        if index < 0:
            # Handle negative index by counting from end of video
            index = self.num_frames + index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if ret:
            return frame[self.mask]
        else:
            self.cap.release()
            raise IndexError(f"Index {index} is out of range for this video")
    
    def show_frame(self, index):
        """Displays the frame at the given index in the video."""
        frame = self[index]
        return Image.fromarray(frame)
    
    def show_video_widget(self):
        """Displays a widget for navigating the video frames."""
        out = widgets.Output()
        frame = self[1]
        with out:
            clear_output(wait=True)
            display(Image.fromarray(frame))
        
        slider = widgets.IntSlider(min=0, max=self.num_frames-1, step=1, value=0, description='Frame:')

        def on_value_change(change):
            index = change['new']
            frame = self[index]
            with out:
                clear_output(wait=True)
                display(Image.fromarray(frame))
                
        slider.observe(on_value_change, names='value')

        vbox = widgets.VBox([out, slider])
        return vbox
    
    def reset_mask(self):
        self.mask=(slice(None), slice(None))
    
    
    @property
    def mask(self):
        """The mask to be applied to each frame of the video."""
        return self._mask
    
    @mask.setter
    def mask(self, value):
        self._mask = value
        
    def find_radius(self, frame, thresh=117):
        """Finds the radius in the given frame.
        
        Args:
            frame (numpy.ndarray): The frame to process.
            thresh (int): The threshold value to use for binarization. Defaults to 117.
        
        Returns:
            int: The radius of the air bubble in the frame.
        """
        frame = frame[self.mask]
        im_th = cv2.threshold(frame, thresh, 255, cv2.THRESH_BINARY)[1]
        h, w = im_th.shape[:2]
        im_floodfill = im_th.copy()
        mask1 = np.zeros((h+2, w+2), np.uint8)
        
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask1, (int(w/2),0), 1);
        cv2.floodFill(im_floodfill, mask1, (int(w/2),h-1), 1);

        airwidth = max(np.sum(mask1, axis=0)[1:-1])
        return h-airwidth+2
    
    def find_radius_all(self, thresh=117):
        """Finds the radius in all the frames.
        
        Args:
            thresh (int): The threshold value to use for binarization. Defaults to 117.
        
        Returns:
            list: tuples with (frame_index, radius)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        radius_list = []
        for i in range(self.num_frames):
            ret, frame = self.cap.read()
            if ret:
                radius = self.find_radius(frame, thresh)
                radius_list.append((i, radius))
            else:
                break
                
        res_table=pd.DataFrame.from_records(radius_list,columns=['frame_num','radius_pixels'])
        res_table['filename']=self.video_filename
        return res_table

class bbox_select():
    def __init__(self,im):
        self.im = im
        self.fig,self.ax = plt.subplots()
        self.img = self.ax.imshow(self.im.copy())

    @property
    def mask(self):
        """The mask to be applied to each frame of the video."""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        mask = (slice(int(ylim[1]), int(ylim[0])),slice(int(xlim[0]), int(xlim[1])))
        return mask
    


def fit_linear_caber(t, r, strike_len_s=0.1, fps=100, surface_tension=40e-3, eta_shear=None,filename=None, show_plot=False):
    """
    Fit a linear model to the experimental data for a capillary breakup experiment and calculate relevant parameters.

    Parameters:
    -----------
    t : array_like
        The time data for the experiment, in seconds.
    r : array_like
        The radius data for the experiment, in meters.
    strike_len_s : float, optional
        The length of the strike (extension) phase of the experiment, in seconds. Default is 0.1.
    fps : float, optional
        The frame rate of the experiment, in frames per second. Default is 100.
    surface_tension : float, optional
        The surface tension of the fluid being used in the experiment, in mN/m. Default is 40.
    eta_shear : float, optional
        The shear viscosity of the fluid being used in the experiment, in Pa s. Default is None.
    filename : str, optional
        If given, the filename will be included in the title of the plot. Default is None.
    show_plot : bool, optional
        If True, a plot of the experimental data and the fit will be shown. Default is False.

    Returns:
    --------
    time_to_breakup : float
        The time taken for the capillary to break up, in seconds.
    R_0 : float
        The initial radius of the capillary, in meters.
    result : lmfit.ModelResult
        The result of the linear fit.
    eta_ext : float
        The extensional viscosity of the fluid being used in the experiment, in Pa s.
    trouton_ratio : float or None
        The Trouton ratio of the fluid being used in the experiment, defined as the ratio of extensional viscosity to shear viscosity. If eta_shear is None, trouton_ratio will be None.
    """
    
    # Find the index where the radius starts to decrease
    start_decrease = np.argmax(np.diff(r) < 0)
    strike_len_frame = int(strike_len_s * fps)
    start_fit = strike_len_frame + start_decrease
    t_exp = t - t[start_fit]

    # Find the index where the radius gets to zero
    zero_index = np.argmin(np.abs(r[start_fit:])) + start_fit
    time_to_breakup = t[zero_index] - t[start_decrease]

    # Create the linear model for fitting
    def linear_rt(t, R_0, sigma_over_eta):
        return 0.0709 * sigma_over_eta * (14.1 * R_0 / sigma_over_eta - t)

    # Create the model and parameters for fitting
    model = lmfit.Model(linear_rt)
    params = model.make_params(R_0=3, sigma_over_eta=1)

    # Perform the fit on the intermediate data
    result = model.fit(r[start_fit:zero_index], params, t=t_exp[start_fit:zero_index])
    R_0_exp = r[start_fit]

    Rdot = -result.params['sigma_over_eta'].value / result.params['R_0'].value *1E-3 * 0.0709
    eta_ext = -surface_tension / 2 / Rdot

    title_parts = [
        f"Time to breakup: {time_to_breakup:.2f} s",
        f"R_0 : {R_0_exp:.2e} m",
        f"fit parameters: sigma_over_eta {result.values['sigma_over_eta']:.4e} m/s, R_0 {result.values['R_0']:.2e} m",
        f"Ext_viscosity: {eta_ext:.2e} Pa s"
    ]

    trouton_ratio = None
    if eta_shear is not None:
        trouton_ratio = eta_ext / eta_shear
        title_parts.append(f"Trouton Ratio: {trouton_ratio:.2e}")

    title_string = '\n'.join(title_parts)
    if filename is not None:
        title_string = f"{filename} \n" + title_string

    if show_plot:
        fig, ax = plt.subplots()
        ax.plot(t_exp, r, 'bo', label='data')
        ax.plot(t_exp[start_fit:zero_index], result.best_fit, 'r-', label='fit')
        ax.axvline(x=t_exp[start_fit], color='k', linestyle='--', label='extension completed')
        ax.axvline(x=t_exp[zero_index], color='g', linestyle='-.', label='breakup')
        ax.set_xlabel('Time')
        ax.set_ylabel('Radius')
        ax.legend()
        ax.set_title(title_string)
        fig.tight_layout()
        fig.savefig('caber_result.jpg')
        plt.show()
        
        
    return {'time_to_breakup':time_to_breakup,
            'Initial radius exp': R_0_exp,
            'lmfit_result':result, 
            'extensional_viscosity_pas': eta_ext, 
            'trouton_ration':trouton_ratio}

def calibrate_res_table(res_table, fps=100, mmperpix=6/454):
    strike_len=0.2*fps
    res_table['strike_len_s']=strike_len
    res_table['fps']=fps
    res_table['t_s']=res_table['frame_num']/fps
    res_table['radius_m']=res_table['radius_pixels']*mmperpix*1E-3
    return res_table