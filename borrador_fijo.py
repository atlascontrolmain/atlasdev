class UsuarioGeneral:
    def __init__(self, usuario, edad):
        self.usuario = usuario
        self.edad = int(edad)

    def mostrarEdad(self):
        return self.edad
    
    def mostrarUsuario(self):
        return f'Hola soy un usuario registrado como: {self.usuario}'


class UsuarioCiego(UsuarioGeneral):
    def __init__(self, usuario, edad, ciego):
        # Llamada directa a UsuarioGeneral en lugar de super()
        UsuarioGeneral.__init__(self, usuario, edad)
        self.ciego = ciego

    def mostrarDiscapacidad(self):
        return f'Usuario ciego: {self.usuario}'

    
class UsuarioTerceraEdad(UsuarioGeneral):
    def __init__(self, usuario, edad, terceraEdad):
        # Llamada directa a UsuarioGeneral en lugar de super()
        UsuarioGeneral.__init__(self, usuario, edad)
        self.terceraEdad = terceraEdad

    def mostrarDiscapacidad(self):
        return f'Usuario registrado como tercera edad: {self.usuario}'


class CiegoTerceraEdad(UsuarioCiego, UsuarioTerceraEdad):
    def __init__(self, usuario, edad, ciego, terceraEdad, usuarioCiegoTerceraEdad):
        UsuarioGeneral.__init__(self, usuario, edad)
        UsuarioCiego.__init__(self, usuario, edad, ciego)
        UsuarioTerceraEdad.__init__(self, usuario, edad, terceraEdad)
        self.usuarioCiegoTerceraEdad = usuarioCiegoTerceraEdad

    def mostrarDiscapacidad(self):
        return f'{UsuarioGeneral.mostrarUsuario(self)}, {self.usuarioCiegoTerceraEdad}'


# Pruebas
usuario1 = UsuarioGeneral('Braulio', 32)
print(usuario1.mostrarUsuario())

usuario2 = UsuarioCiego('Braulio', 22, 'Ciego')
print(usuario2.mostrarDiscapacidad())

usuario3 = UsuarioTerceraEdad('Braulio', 92, 'Tercera edad')
print(usuario3.mostrarDiscapacidad())

usuario4 = CiegoTerceraEdad('Braulio', 92, 'Ciego', 'Tercera edad', '2 comorbilidades')
print(usuario4.mostrarDiscapacidad())