# herencia multiple

class UsuarioGeneral:
    def __init__(self, usuario, edad):
        self.usuario = usuario #nombre del usuario
        self.edad = int (edad)

    def mostrarEdad (self):
        return self.edad
    
    def mostrarUsuario (self):
        return f'Hola soy un usuario registrado como: {self.usuario}'

class UsuarioCiego(UsuarioGeneral):
    def __init__(self, usuario, edad, ciego):
        super().__init__(usuario, edad) # aqui esta llamando a la clase padre
        self.ciego = ciego #requiere tres argumentos

    def mostrarDiscapacidad (self):
        return 'Usuario ciego'

class UsuarioTerceraEdad(UsuarioGeneral):
    def __init__(self):
        pass

class CiegoTerceraEdad (UsuarioCiego, UsuarioTerceraEdad):
    def __init__(self):
        pass

usuario1 = UsuarioGeneral ('Braulio', 32)
print (usuario1.mostrarUsuario())

usuario2 = UsuarioCiego('Braulio', 22, 'Ciego')
print (usuario2.mostrarDiscapacidad())