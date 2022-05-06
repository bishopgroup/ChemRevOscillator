import sys 

try:
    jmax=int(sys.argv[1]) #changing this requires full recompilation and symbolic derivatives.
except:
    jmax=100


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import time
import shutil
import imp

import numpy as np
from pyodesys.symbolic import SymbolicSys
from pyodesys.native import native_sys




def install():
    tmax = 10
    kp = 1
    kn = 1
    ka = 0.5
    kd1 = 1
    kd2 = 1
    kd3 = 1
    kd = 10
    m0 = 1000
    nc=4
    e=0
    ec=1.5
    kp=1

    print("Computing analytical derivatives ...")
    neq2_isodesic = SymbolicSys.from_callback(neq2_isodesic_rhs, ny=jmax + 1, nparams=6, backend="sympysymengine")
    neq_isodesic = SymbolicSys.from_callback(neq_isodesic_rhs, ny=jmax + 1, nparams=4, backend="sympysymengine")
    eq_isodesic = SymbolicSys.from_callback(eq_isodesic_rhs, ny=jmax, nparams=2, backend="sympysymengine")
    eq_coop = SymbolicSys.from_callback(eq_coop_rhs, ny=jmax, nparams=4, backend="sympysymengine")

    print("Trying to compile ...")
    eq_coop_compiled= native_sys['cvode'].from_other(eq_coop)
    ic = np.zeros(jmax)
    ic[0] = m0
    eq_coop_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(nc,e,ec,kp), integrator="cvode")

    neq2_isodesic_compiled = native_sys['cvode'].from_other(neq2_isodesic)
    ic = np.zeros(jmax + 1)
    ic[0] = m0
    neq2_isodesic_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(kp, kn, ka, kd1,kd2,kd3), integrator="cvode")

    neq_isodesic_compiled = native_sys['cvode'].from_other(neq_isodesic)
    ic = np.zeros(jmax + 1)
    ic[0] = m0
    neq_isodesic_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(kp, kn, ka, kd), integrator="cvode")

    eq_isodesic_compiled = native_sys['cvode'].from_other(eq_isodesic)
    ic = np.zeros(jmax)
    ic[0] = m0
    eq_isodesic_compiled.integrate(np.linspace(0, tmax, 1000), ic, params=(kp, kn), integrator="cvode")
    print("Compilation successful, saving systems!")
    shutil.copytree(eq_coop_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/eq_coop".format(jmax) + "/mod")
    shutil.copytree(neq2_isodesic_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/neq2_isodesic".format(jmax) + "/mod")
    shutil.copytree(neq_isodesic_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/neq_isodesic".format(jmax) + "/mod")
    shutil.copytree(eq_isodesic_compiled._native.mod._binary_path.rpartition("/")[0],
                    "compiled/{}/eq_isodesic".format(jmax) + "/mod")


class results_wrapper():
    def __init__(self,xout,yout,stats=None):
        self.xout=xout
        self.yout=yout
        self.stats=stats
    pass


###
class base_integration_wrapper():
    def __init__(self, name):
        import imp

        fobj, filename, data = imp.find_module("_cvode_wrapper",
                                           ["compiled/{}/".format(jmax)+name+"/mod"])
        self.__mod = imp.load_module('_cvode_wrapper', fobj, filename, data)

    def integrate(self, teval, C0, params, integrator='cvode',
                  nsteps=10000, atol=1e-08, rtol=1e-08):

        C0 = np.atleast_2d(C0)
        intern_x = teval  ##I think this needs to be a linspace
        intern_x = np.atleast_2d(np.ascontiguousarray(intern_x, dtype=np.float64))
        P = np.array(np.atleast_2d(params), dtype=np.float64)
        atol = np.array([atol] * len(C0))
        yout, stats = self.__mod.integrate_predefined(y0=np.atleast_2d(C0), xout=np.atleast_2d(
            np.ascontiguousarray(intern_x, dtype=np.float64)),
                                                      params=P, atol=atol,
                                                      rtol=rtol,
                                                     mxsteps=nsteps, dx0=None)

        return results_wrapper(np.squeeze(intern_x), yout[0], stats)



def eq_isodesic_rhs(t, C, p, backend=np):
    kp = p[0]
    kn = p[1]

    dC = np.zeros_like(C) * C  ## 
    for n in range(0, jmax):
        dC[n] = -2 * kp * C[n] * sum(C) + kp * sum(C[:n] * (C[:n][::-1])) + 2 * kn * sum(C[n+1:]) - kn * (n) * C[n]

    return dC
def neq_isodesic_rhs(t, C, p, backend=np):
    kp = p[0]
    kn = p[1]
    ka = p[2]
    kd = p[3]

    d = C[0]

    dC = np.zeros_like(C) * C  ## 
    idx = np.arange(jmax + 1)  #

    dC[0] = -ka * d + kd * sum(idx * C)  # one extra flop for the first element which is zero
    dC[1] = ka * d - 2 * kp * C[1] * sum(C[1:]) + 2 * kn * sum(C[2:]) - kd * C[1] + 2 * kd * sum(C[2:])
    for n in range(2, jmax + 1):
        dC[n] = -2 * kp * C[n] * sum(C[1:]) + kp * sum(C[1:n] * (C[1:n][::-1])) + 2 * kn * sum(C[n + 1:])\
                - kn * ( n - 1) * C[n] - n * kd * C[n] + 2 * kd * sum(C[n + 1:])
    return dC

def neq2_isodesic_rhs(t, C, p, backend=np):
    kp = p[0]
    kn = p[1]
    ka = p[2]
    kd1 = p[3]
    kd2 = p[4]
    kd3 = p[5]

    d = C[0]

    dC = np.zeros_like(C) * C  ## 

    idx = np.arange(jmax + 1)

    dC[0] = - ka*d + kd1*C[1] + 2*kd2*sum(C[2:]) + kd3*sum(idx[1:-2]*C[3:])
    dC[1] = -2*kp*C[1]*sum(C[1:]) + 2*kn*sum(C[2:]) - kd1*C[1] + 2*kd2*C[2] + 2*kd3*sum(C[3:]) + ka*d
    for n in range(2, jmax):
        dC[n] =  -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))  - kn * (n-1) * C[n]  + 2*kn*sum(C[n+1:]) \
                 - 2*kd2*(C[n]-C[n+1])  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:]) #using that sum of empty slices is 0
    n = jmax
    dC[n] =  -2*kp*C[n]*sum (C[1:]) + kp*sum(C[1:n]*(C[1:n][::-1]))  - kn * (n-1) * C[n]  + 2*kn*sum(C[n+1:]) \
             - 2*kd2*(C[n]-0)  - kd3*(n-2)*C[n] + 2*kd3*sum(C[n+2:])

    return dC

def eq_coop_rhs(t, C, p, backend=np):

    nc = 4# p[0]
    ec = p[1]
    e = p[2]
    kp = p[3]



    dC = np.zeros_like(C) * C  ## 
    u0 = np.zeros(jmax**2) *e #

    #u0[:nc] = np.arange(nc) * ec  # <= nc, e.g. if nc = 2, this will get points 0 and 1, aka 1 and 2. arnage for nc-1
    #u0[nc:] = np.arange(1, jmax ** 2 + 1 - nc) * e + (
    #            nc - 1) * ec  # for n = 3 at nc=2, this will give a value of 1, with a maximum value of jmax-nc

    for n in range(jmax**2):
        if n < nc:#<= nc, e.g. if nc = 2, this will get points 0 and 1, aka 1 and 2. arnage for nc-1
            u0[n] = n*ec
        else:
            u0[n] = n * e + (
                        nc - 1) * ec  # for n = 3 at nc=2, this will give a value of 1, with a maximum value of jmax-nc


    kn = np.zeros((jmax, jmax))*e
    idx = np.arange(jmax) #j, but stritctly for indexing! since it's offset by 1.

    for i in range(jmax):
        for j in range(jmax):
            kn[i, j] = kp * backend.exp(u0[i + j] - u0[i] - u0[j])
    for n in range(jmax):
        dC[n] = -2*kp*C[n]*sum(C) + kp*sum(C[:n] * (C[:n][::-1])) - C[n]*sum(kn[idx[:n][::-1],idx[:n]]) + 2*sum(kn[n,0:-1-n]*C[n+1:])
    return dC




def load_systems():
    eq_isodesic = base_integration_wrapper("eq_isodesic")
    neq_isodesic = base_integration_wrapper("neq_isodesic")
    neq2_isodesic = base_integration_wrapper("neq2_isodesic")
    eq_coop = base_integration_wrapper("eq_coop")
    return eq_isodesic,neq_isodesic,neq2_isodesic,eq_coop





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    "Max % Conservation Error: ",
    html.Span(id="cons_err",children=""),
    dcc.RadioItems(
        id='model',
        options=[
            {'label': 'Equilibrium Isodesmic', 'value': 'eq'},
            {'label': 'Nonequilibrium Isodesmic', 'value': 'noneq'},
            {'label': 'Nonequilibrium II Isodesmic', 'value': 'noneq2'},
            {'label': 'Equilibrium Cooperative', 'value': 'coopeq'}

        ],
        value='noneq'
    ),
    "m0",
    dcc.Slider(
        id='m0',
        min=0,
        max=1000,
        value=50,
        step=0.25,
        tooltip = { 'always_visible': True }
    ),
    html.Content([
    "tmax",
    dcc.Slider(
        id='tmax',
        min=-2,
        max=2,
        marks={-2: "1/100", -1: "1/10", 0: "1", 1: "10", 2: "100"},
        value=0,
        step=0.1
    ),
    "n points",
    dcc.Slider(
        id='10npoints',
        min=1,
        max=5,
        marks={1: "10", 2: "100", 3: "1000", 4: "10000", 5: "100000"},
        value=3,
        step=0.1
    )]),
    html.Content(["kp",
    dcc.Slider(
        id='kp',
        min=0,
        max=10,
        value=1,
        step=0.025,tooltip = { 'always_visible': True }
    ), "kn",
    dcc.Slider(
        id='kn',
        min=0,
        max=10,
        value=0.5,
        step=0.025, tooltip = { 'always_visible': True }
    ),
    html.Content(children=["ka",
    dcc.Slider(
        id='ka',
        min=0,
        max=10,
        value=1,
        step=0.025,tooltip = { 'always_visible': True }
    )], id="ka_frame"),

    html.Content( children=["kd",dcc.Slider(
            id='kd',
            min=0,
            max=10,
            value=1,
            step=0.025, tooltip = { 'always_visible': True }
    )], id="kd_frame"),
    html.Content(
        children=["kd1",dcc.Slider(
            id='kd1',
            min=0,
            max=10,
            value=1,
            step=0.025, tooltip = { 'always_visible': True }
    ),"kd2",dcc.Slider(
            id='kd2',
            min=0,
            max=10,
            value=1,
            step=0.025, tooltip = { 'always_visible': True }
    ),"kd3",dcc.Slider(
            id='kd3',
            min=0,
            max=10,
            value=1,
            step=0.025, tooltip = { 'always_visible': True }
    )],
        id="kd123_frame")],id="all_iso"),
    html.Content(
        children=["kp", dcc.Slider(
            id='kpc',
            min=0,
            max=10,
            value=1,
            step=0.025, tooltip={'always_visible': True}
        ), "e", dcc.Slider(
            id='eps',
            min=0,
            max=10,
            value=0,
            step=0.025, tooltip={'always_visible': True}
        ), "ec", dcc.Slider(
            id='ec',
            min=0,
            max=10,
            value=1,
            step=0.025, tooltip={'always_visible': True}
        )],
        id="coop_frame")
])

fig = go.Figure(layout={'uirevision':1}) #no update on par. change


@app.callback(
    [Output('graph-with-slider', 'figure'),Output('cons_err', 'children')],
    [Input('model', 'value'),Input('tmax', 'value'),
     Input('10npoints', 'value'),
     Input('kp', 'value'),Input('kn', 'value'),
     Input('ka', 'value'),Input('kd', 'value'),
     Input('m0', 'value'),Input('kd1','value'),
     Input('kd2','value'),Input('kd3','value'),
     Input('kpc','value'), Input('eps','value'),Input('ec','value')])

def update_figure(model,logtmax,lognpoints,kp,kn,ka,kd,m0,kd1,kd2,kd3,kpc,e,ec):
    start=time.perf_counter()
    tmax = 10**logtmax
    npoints=int(10**lognpoints)
    err=-1
    if model == "eq":
        ic = np.zeros(jmax)
        ic[0] = m0
        res = eq_isodesic_compiled.integrate(np.linspace(0, tmax, npoints), ic, params=(kp, kn), integrator="cvode")
        fig.data = []
        for idx,trace in enumerate(res.yout.T):

            fig.add_trace(go.Scatter(
                x=res.xout.copy(),
                y=trace.copy(),
                name="length {}".format(idx+1)
            ))
        err=max(np.abs(m0-np.sum(res.yout.T * np.arange(1,jmax + 1)[:,None],axis=0)))/m0*100

    elif model=="noneq":
        ic = np.zeros(jmax+1)
        ic[0] = m0
        res = neq_isodesic_compiled.integrate(np.linspace(0, tmax, npoints), ic, params=(kp, kn,ka,kd), integrator="cvode")
        fig.data = []
        fig.add_trace(go.Scatter(
            x=res.xout,
            y=res.yout.T[0],
            name="d"
        ))
        for idx, trace in enumerate(res.yout.T[1:]):
            fig.add_trace(go.Scatter(
                x=res.xout,
                y=trace,
                name="length {}".format(idx+1)
            ))
        idx= np.arange(0,jmax + 1)
        idx[0] = 1
        err=max(np.abs(m0-np.sum(res.yout.T *idx[:,None],axis=0)))/m0*100

    elif model=="noneq2":
        ic = np.zeros(jmax+1)
        ic[0] = m0
        res = neq2_isodesic_compiled.integrate(np.linspace(0, tmax, npoints), ic, params=(kp, kn,ka,kd1,kd2,kd3), integrator="cvode")
        fig.data = []
        fig.add_trace(go.Scatter(
            x=res.xout,
            y=res.yout.T[0],
            name="d"
        ))
        for idx, trace in enumerate(res.yout.T[1:]):
            fig.add_trace(go.Scatter(
                x=res.xout,
                y=trace,
                name="length {}".format(idx+1)
            ))
        idx= np.arange(0,jmax + 1)
        idx[0] = 1
        err=max(np.abs(m0-np.sum(res.yout.T *idx[:,None],axis=0)))/m0*100

    elif model == "coopeq":
        ic = np.zeros(jmax)
        ic[0] = m0
        res = eq_coop_compiled.integrate(np.linspace(0, tmax, npoints), ic, params=(0,kpc,e,ec),
                                               integrator="cvode")
        fig.data = []
        fig.add_trace(go.Scatter(
            x=res.xout,
            y=res.yout.T[0],
            name="d"
        ))
        fig.data = []
        for idx,trace in enumerate(res.yout.T):

            fig.add_trace(go.Scatter(
                x=res.xout.copy(),
                y=trace.copy(),
                name="length {}".format(idx+1)
            ))
        err=max(np.abs(m0-np.sum(res.yout.T * np.arange(1,jmax + 1)[:,None],axis=0)))/m0*100
        # print(diff)
    fig.update_layout( xaxis={"title":"Time"},yaxis={"title":"Concentration"})

    return fig,"{:.1e}, Evaluation time: {:.1e}s".format(err,time.perf_counter()-start)


@app.callback(
    [Output(component_id='ka_frame', component_property='hidden'),
     Output(component_id='kd_frame', component_property='hidden'),
     Output(component_id='kd123_frame', component_property='hidden'),
     Output(component_id='all_iso', component_property='hidden'),
     Output(component_id='coop_frame', component_property='hidden')

     ],
    [Input('model', 'value')])

def select_model(model):
    if model == "eq":
        return [ True,True,True,False,True]
    elif model == "noneq":
        return  [False,False,True,False,True]
    elif model == "noneq2":
        return [False,True,False,False,True]
    elif model == "coopeq":
        return [False, True, False, True, False]

if __name__ == '__main__':
    import os
    if os.path.isdir("compiled/{}/eq_isodesic".format(jmax)):
        print("> Looks like you've already installed everything, you're good to go.")
        print("> To recompile, delete all the folders in the 'compiled' directory")
    else:
        print("> Setting everything up, this can take ~10minutes")
        install()

    eq_isodesic_compiled, neq_isodesic_compiled,neq2_isodesic_compiled,eq_coop_compiled = load_systems()
    
    server=True
    try:
        a=sys.argv[2]
        server=False
    except:
        pass
    if server:
        app.run_server(host='0.0.0.0')
