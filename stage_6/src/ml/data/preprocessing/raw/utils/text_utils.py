# pylint: disable=R0801
"""
Module with utils for processing text data
"""

import numpy as np
import numpy.typing as npt


POPULAR_COLORS = np.array(
    [
        "бежевый",
        "белый",
        "голубой",
        "желтый",
        "зеленый",
        "коричневый",
        "красный",
        "оранжевый",
        "розовый",
        "серый",
        "синий",
        "фиолетовый",
        "черный",
        "другой",
    ]
)

color_dict = {color: i + 1 for i, color in enumerate(POPULAR_COLORS)}

color_dict[""] = len(POPULAR_COLORS) + 10
color_dict["[]"] = len(POPULAR_COLORS) + 10
color_dict["['']"] = len(POPULAR_COLORS) + 10
color_dict['[""]'] = len(POPULAR_COLORS) + 10


def which_color(x: str):
    """
    Determines the color category of a given string or list of strings based
        on predefined popular colors.

    Args:
        x: The input, which can be a string or a list of strings,
            containing color-related information.

    Returns:
        The color category (value from the `color_dict`) that matches the input, or the default
        color "другой" if no match is found or an exception occurs.
    """
    # using try\except
    # for case if x is not iterable
    # or has other type
    try:
        if isinstance(x, str):
            for c in POPULAR_COLORS:
                if c in x:
                    return color_dict[c]
            return color_dict["другой"]

        for c in POPULAR_COLORS:
            for xx in x:
                if c in xx:
                    return color_dict[c]
        return color_dict["другой"]

    except Exception:
        return color_dict["другой"]


# Словарь для парсинга колонки characteristics
chars_dict = {
    "длина юбки/платья": "length",
    "длина юбки\\платья": "length",
    "модель платья": "model",
    "назначение": "purpose",
    "покрой": "cut",
    "рисунок": "pattern",
    "тип карманов": "pocket",
    "тип ростовки": "height",
    "тип рукава": "sleeve",
    "состав": "material",
    "вид застежки": "closure",
    "вид застёжки": "closure",
    "вырез горловины": "neckline",
    "страна производства": "country",
}

# ---------------------------------------------------
# Словари для проверки подстрок и составления таблицы
# ---------------------------------------------------

# Застежка
closure_dict = {
    "без застежки": ["без", "нет"],
    "молния": ["молни"],
    "пуговица": ["пуговиц"],
    "завязка": ["завязк"],
    "пояс": ["пояс"],
    "шнуровка": ["шнур"],
    "резинка": ["резин"],
}

# Рисунок
pattern_dict = {
    "абстракция": ["абстрак"],
    "без рисунка": ["без", "однотон", "нет"],
    "горох": ["горох", "горош"],
    "клетка": ["клет"],
    "леопардовый": ["леопард"],
    "полоска": ["полос"],
    "фигуры": ["фигур", "геометр"],
    "цветы": ["цвет", "растен"],
}

# Назначение
purpose_dict = {
    "большие размеры": ["больш"],
    "вечернее": ["вечер"],
    "выпускной": ["выпуск"],
    "беремен": ["для беременн", "будущие мамы", "род"],
    "кормления": ["корм"],
    "крещения": ["крещ"],
    "домашнее": ["дом"],
    "повседневная": ["повседнев"],
    "свадьба": ["свадьб"],
    "пляж": ["пляж"],
    "новый год": ["новый"],
    "школа": ["школ"],
    "спорт": ["спорт"],
    "офис": ["офис"],
}

# Карманы
pocket_dict = {
    "в_шве": ["в шв", "бок"],
    "без_карманов": ["без", "нет"],
    "прорезные": ["прорез"],
    "потайные": ["тайн"],
    "накладные": ["наклад"],
}

# Рукава
sleeve_dict = {
    "без рукавов": ["без", "нет"],
    "длинные": ["дли"],
    "короткие": ["кор"],
    "3/4": ["3/4", "3\\4", "34", "3", "4"],
    "7/8": ["7/8", "7\\8", "78", "7", "8"],
}

# Длина
length_dict = {
    "миди": [
        "миди",
        "серед",
        "10",
        "11",
        "ниже",
        "по",
    ],
    "макси": [
        "макси",
        "длин",
        "12",
        "13",
        "14",
        "15",
        "16",
        "в пол",
        "пол",
    ],
    "мини": [
        "мини",
        "кор",
        "9",
        "8",
        "до",
        "выше",
        "корот",
    ],
}


# Модель
model_dict = {
    "футляр": ["футл"],
    "рубашка": ["рубаш"],
    "открытое": ["откр"],
    "запах": ["запах"],
    "прямое": ["прям"],
    "кожаное": ["кожа"],
    "свадьба": ["свад"],
    "лапша": ["лап"],
    "вязаное": ["вяз"],
    "комбинация": ["комб"],
    "футболка": ["футб"],
    "водолазка": ["водо"],
    "бохо": ["бохо"],
    "сарафан": ["сараф"],
    "пиджак": ["пидж"],
    "трапеция": ["трапец"],
    "мини": ["мини"],
    "макси": ["макси"],
    "миди": ["миди"],
    "свободное": ["свобод"],
    "а-силуэт": [
        "а-силуэт",
        "а- силуэт",
        "а -силуэт",
        "а силуэт",
        "асилуэт",
        "a-силуэт",
        "a- силуэт",
        "a -силуэт",
        "a силуэт",
        "aсилуэт",
    ],
    "туника": ["тун"],
    "приталеное": ["тален"],
    "поло": ["поло"],
    "парео": ["парео"],
}


# Покрой
cut_dict = {
    "асимметричный": ["асимметр"],
    "приталенный": ["притален"],
    "рубашечный": ["рубаш"],
    "свободный": ["свободн"],
    "укороченный": ["укороч"],
    "удлиненный": ["удлин"],
    "облегающий": ["облега"],
    "полуприлегающий": ["полуприлег"],
    "прямой": ["прям"],
    "а-силуэт": [
        "а-силуэт",
        "а силуэт",
        "а- силуэт",
        "асилуэт",
        "a-силуэт",
        "a силуэт",
        "a- силуэт",
        "aсилуэт",
    ],
    "оверсайз": ["овер", "over"],
    "трапеция": ["трапец"],
    "длинный": ["длин"],
}

# Тип ростовки
height_dict = {
    "для высоких": [
        "1,7",
        "1,8",
        "1,9",
        "2,0",
        "17",
        "18",
        "19",
        "20",
        "для выс",
        "длявыс",
    ],
    "для невысоких": [
        "1,2",
        "1,3",
        "1,4",
        "1,5",
        "12",
        "13",
        "14",
        "15",
        "невыс",
        "не выс",
        "для невыс",
        "для не выс",
        "дляневыс",
        "дляне выс",
    ],
    "для среднего роста": [
        "для средн",
        "длясредн",
        "1,6",
        "16",
        "средн",
    ],
    "для всех": [
        "для всех",
        "длявсех",
        "безогр",
        "без огр",
    ],
}

# Материал
material_dict = {
    "акрил": ["акрил"],
    "бамбук": ["бамбук"],
    "вискоза": ["вискоза"],
    "кашемир": ["кашемир"],
    "кожа": ["кожа"],
    "лайкра": ["лайкра"],
    "лен": ["лен"],
    "нейлон": ["нейлон"],
    "полиамид": ["полиамид"],
    "полиэстер": ["полиэстер"],
    "спандекс": ["спандекс"],
    "трикотаж": ["трикотаж"],
    "шерсть": ["шерсть"],
    "шелк": ["шелк"],
    "штапель": ["штапель"],
    "шифон": ["шифон"],
    "хлопок": ["хлопок"],
    "эластан": ["эластан"],
}

# Вырез
neckline_dict = {
    "круглый": ["круг"],
    "классический": ["классич"],
    "стойка": ["стойк"],
    "овал": ["овал"],
    "бретели": ["бретел"],
    "v-образный": ["v"],
    "сердечко": ["сердеч"],
    "американка": ["америк"],
    "фигурный": ["фигур"],
    "u-образный": ["u"],
    "гольф": ["гольф"],
    "хомут": ["хомут"],
    "лодочка": ["лодоч"],
    "отложной": ["отлож"],
    "кармен": ["кармен"],
    "бант": ["бант"],
    "капюшон": ["капюш"],
    "квадратный": ["квадра"],
    "открытый": ["откр", "спина", "плеч"],
    "с горлом": ["с горлом", "горло"],
}

# Страна
country_dict = {
    "Россия": ["россия", "russia"],
    "Беларусь": ["беларусь"],
    "Турция": ["турция"],
    "Франция": ["франция"],
    "Киргизия": ["киргизия"],
    "Китай": ["китай", "china"],
    "Италия": ["италия"],
    "Индия": ["индия"],
    "Бангладеш": ["бангладеш"],
    "Узбекистан": ["узбекистан"],
    "Вьетнам": ["вьетнам"],
    "Гонконг": ["гонконг"],
}


chars_formating_dicts = {
    "length": length_dict,
    "model": model_dict,
    "purpose": purpose_dict,
    "cut": cut_dict,
    "pattern": pattern_dict,
    "pocket": pocket_dict,
    "height": height_dict,
    "sleeve": sleeve_dict,
    "material": material_dict,
    "closure": closure_dict,
    "neckline": neckline_dict,
    "country": country_dict,
}


def parse_char_dict(characteristics: dict):
    """
    Parses and filters item characteristics, extracting relevant information
    based on predefined names.

    Args:
        characteristics: A dictionary containing item characteristics, where each key is
            a characteristic name, and value is a list of associated values.

    Returns:
        A new dictionary containing filtered characteristics names as keys,
        with lowercase char values as a list of strings as values.
    """
    filtered_chars = {}
    for char in characteristics:
        try:
            filtered_chars[chars_dict[str.lower(char["charcName"])]] = [
                str.lower(x) for x in char["charcValues"]
            ]
        except Exception:
            pass
    return filtered_chars


def get_char_value(characteristics: dict, char: str):
    """
    Retrieves the value associated with a specific characteristic
    from a dictionary of characteristics.

    This function attempts to retrieve a specific characteristic (e.g., color) from a dictionary of
    item characteristics. If the characteristic is found, its associated value (which is expected to
    be a list) is returned. If the characteristic is not found, a default list with single "unknown"
    value is returned.

    Args:
        characteristics: A dictionary containing item characteristics, where each key is a
            characteristic name (string) and the value is the associated value (list of strings).
        char: The name of the characteristic to retrieve.

    Returns:
        The value associated with the given characteristic (list of strings),
        or ["unknown"] if not found.
    """
    try:
        return characteristics[char]
    except Exception:
        return ["unknown"]


def format_chars(chars: npt.ArrayLike, char_dict: dict):
    """
    Formats item characteristics based on predefined character types and values.

    This function takes a list or array-like object of strings and a character type dictionary.
    It iterates through the dictionary and checks whether any of the values for a given character
    type are present in the input array. If a match is found, the type of the matched value is
    returned. If not match is found, the default "unknown" value is returned.

    Args:
        chars: A NumPy array-like of strings representing the characteristics of an item.
        char_dict: A dictionary where keys are character types and values are lists
            of possible values for that character type.

    Returns:
        The matched char type if found in array, or "unknown" if no match is found.
    """

    for char_type, char_values in char_dict.items():
        if any(v in "".join(chars) for v in char_values):
            return char_type

    return "unknown"
