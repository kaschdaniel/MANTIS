# PROCESSING

def beamsplitter():
    return (1/np.sqrt(2)) * np.array([[1, 1j],
                                    [1j, 1]], dtype=np.complex128) #Datatype not optimal for GPU Acceleration

def 