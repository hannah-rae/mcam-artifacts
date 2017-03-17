# Should be run from /Users/hannahrae/src/mcam-artifacts/v2
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
import numpy as np
from glob import glob

d = 'test_images/test2' # TODO: Make this an argument
fnames = glob(d + '/*')
fn_dict = {}
for f in fnames:
    num = f[18:].split('_')[0]
    fn_dict[num] = f

vals = np.empty((7,8))
for i, row in enumerate(vals):
    for j, col in enumerate(row):
        n = str(i*8 + j)
        f = fn_dict[n]
        p = f.split('_')[4][:-4]
        vals[i,j] = p


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Probability of Quality Acceptance Across Image')
plotly_fig = tls.mpl_to_plotly(fig)
trace = dict(z=np.flipud(vals), type="heatmap", zmin=0.0, zmax=1.0, colorscale='YlOrRd')
plotly_fig['data'] = [trace]
plotly_fig['layout']['xaxis'].update({'autorange':True, 'tickmode':'array', 'ticklen':0, 'tickvals':[0,1,2,3,4,5,6,7]})
plotly_fig['layout']['yaxis'].update({'autorange':True, 'tickmode':'array', 'ticklen':0, 'tickvals':[0,1,2,3,4,5,6]})
plot_url = py.plot(plotly_fig, filename='mpl-basic-heatmap')