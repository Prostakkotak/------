from gensim.models import Word2Vec
import re
import numpy as np

bd = {
    -0.1097786: "Корпоративный портал — это веб-интерфейс для доступа сотрудника к корпоративным данным и приложениям Дашборд - это интерактивная аналитическая панель, графический интерфейс. Смысл в том, что на одном экране расположены все ключевые метрики, показатели цели или процессов. С помощью этих метрик можно выявить и проанализировать тренды и изменения Корпоративная информационная системка - это комплекс программных и аппаратных средств, предназначенных для сбора, хранения, обработки и передачи информации внутри организации",
    0.05530073: "Веб-приложение - это программное приложение, которое работает в веб-браузере пользователя и доступно через интернет. Веб-приложения выполняют различные функции, включая обработку данных, интерактивные задачи, передачу информации и другие операции. Они обычно разрабатываются с использованием веб-технологий и могут быть использованы на разных устройствах с доступом к интернету. Фреймворк - это структура, набор инструментов, библиотек и правил, которые облегчают разработку программного обеспечения Библиотека - это совокупность программных модулей, функций или классов, предназначенных для решения определенных задач в рамках программирования",
    -0.033034403: "Информационная безопасность - это область, занимающаяся защитой информации от несанкционированного доступа, использования, раскрытия, изменения или уничтожения Кибератаки - злонамеренные действия, направленные на компьютерные системы, сети, программное обеспечение или данные с целью нарушения их работы, получения несанкционированного доступа, кражи информации, разрушения или иного вредного воздействия на цифровую инфраструктуру Сертифицированное программное обеспечение - это программное обеспечение, прошедшее определенные проверки, тестирования и процедуры оценки, чтобы подтвердить соответствие определенным стандартам, нормативам или требованиям"
}

w2v_model = Word2Vec.load('resources/vaskovsky.model')

def get_closest_category_vector(query_vector):
    closest_category = None
    min_distance = float('inf')

    for category, value_text in bd.items():
        # Преобразование текста в вектор
        value_vector = get_average_vector(value_text)

        if value_vector is not None:
            distance = np.linalg.norm(query_vector - value_vector)
            if distance < min_distance:
                min_distance = distance
                closest_category = category

    return closest_category

# Функция для получения среднеарифметического значения вектора запроса
def get_average_vector(query):
    # Разбивка запроса на массив слов
    words = query.split()

    # Инициализация массива векторов слов
    word_vectors = []

    # Получение векторов для каждого слова в запросе
    for word in words:
        # Проверка, что слово имеет векторное представление
        if word in w2v_model.wv:
            word_vectors.append(w2v_model.wv[word])

    # Рассчет среднеарифметического значения векторов
    if word_vectors:
        average_vector = np.mean(word_vectors, axis=0)
        return average_vector
    else:
        return None

user_query = input("Введите ваш запрос: ")
query_vector = get_average_vector(user_query)

if query_vector is not None:
    closest_category = get_closest_category_vector(query_vector)
    print(bd[closest_category])
else:
    print("Для некоторых слов в запросе нет векторных представлений.")