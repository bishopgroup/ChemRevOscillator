FROM bjodah/bjodahimg18ff:v1.8
RUN pip3 install dash smt pathos h5py 
COPY chem_rev_tool.py /opt/chem_rev_tool.py 
RUN python3 /opt/chem_rev_tool.py 25 nosrv
RUN python3 /opt/chem_rev_tool.py 50 nosrv
RUN python3 /opt/chem_rev_tool.py 100 nosrv
RUN python3 /opt/chem_rev_tool.py 150 nosrv

CMD python3 /opt/chem_rev_tool.py
