from .base import BaseProgram


class IIS3Program(BaseProgram):
    """
    ИИС-3: нет софинансирования, нет пенсии (обычно life_table=None),
    но логика взносов и накоплений такая же.

    Если в будущем появятся особенности (например, лимит на взносы, другой tax deduction),
    их можно будет добавить здесь.
    """
    pass