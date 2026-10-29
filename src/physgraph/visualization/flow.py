"""Export numerical periodic-fluid frames and passive tracers as an animation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_flow(result, path, fps=15):
    times=result['times']
    if len(times)<2 or not np.allclose(np.diff(times),np.diff(times)[0]):
        raise ValueError('GIF export requires uniformly spaced physical frame times')
    fig,ax=plt.subplots(figsize=(6,6))
    limit=max(float(np.max(np.abs(result['vorticity']))),1e-12)
    picture=ax.imshow(result['vorticity'][0],origin='lower',extent=(0,2*np.pi,0,2*np.pi),
                      cmap='coolwarm',vmin=-limit,vmax=limit,interpolation='bilinear')
    dots=ax.scatter(*result['particles'][0].T,s=10,c='black')
    ax.set(xlabel='x',ylabel='y',xlim=(0,2*np.pi),ylim=(0,2*np.pi),aspect='equal')
    fig.colorbar(picture,ax=ax,label='Numerical vorticity')
    title=ax.set_title('')
    def update(i):
        picture.set_data(result['vorticity'][i])
        dots.set_offsets(result['particles'][i])
        title.set_text(f'Periodic flow + passive tracers | t={times[i]:.2f}')
        return picture,dots,title
    try:
        animation=FuncAnimation(fig,update,frames=len(times),interval=1000/fps)
        animation.save(path,writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
