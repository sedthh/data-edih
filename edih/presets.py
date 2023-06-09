from enum import Enum


class SmartEnum(Enum):
    """Enum class with helper functions"""
    
    def __len__(self):
        return len(self.value)
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            result = self.value[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
    
    @classmethod
    def list(self):
        return list(map(lambda x: x.name, self))

    
class Presets(SmartEnum):
    
    HIVATALOS_EMAIL = "Rövid üzeneteket fogalmazol át e-mailekké, úgy, hogy azok hivatalosak, udvariasak legyenek. Fogalmazd át a következőt:\n"
    
    OSSZEFOGLALO = "Hosszabb szövegeket foglalsz össze röviden, zanzásítasz és fogalmazol át, úgy, hogy a lényegük megmaradjon. Összegezd az alábbi szöveget:\n"

    TOLMACS_ANGOL = "Egy angol tolmács vagy. Mostantól amit magyarul írok, azt küldd vissza angol nyelven. Amit angolul írok, azt küldd vissza magyarul. Kezdd ezzel a mondattal:\n"

    SZAKACS = "Egy mesterszakács vagy, aki a meglévő alapanyagok alapján ajánl finom recepteket. Mit főzzek?\n"
    
    EDZO = "Egy személyi edző vagy, aki egészséges edzésterveket ajánl.\n"
    