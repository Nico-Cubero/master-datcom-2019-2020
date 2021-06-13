# -*- coding: utf-8 -*-
######################################################################################
# Autor: Nicolás Cubero
#
# Descripción: Utilidad para extraer tweets de Twitter sobre la actuación del gobierno
# ante la crisis del coronavirus y almacenarlos en un fichero csv
# Extrae todos los tweets enviados desde el 1 de Abril de 2020 hasta el
# 1 de Mayo de 2020 (inclusive) escritos en castellano y enviados en el territorio peninsular
# (toda localización comprendida a 600 Km de radio desde Madrid) que contengan alguna
# de las combinaciones de términos recogidas.
#
# Nota: Se requiere hacer uso de la librería TWINT para poder hacer
# uso de este script: https://pypi.org/project/twint/
######################################################################################

# Importación de librerías
import twint

# Configurar búsqueda
c = twint.Config()

c.Limit = 100000

# Tweets sólo en español y en España
c.Lang = 'es'
c.Geo = '40.41888888888889,3.6919444444444447,600km'

c.Custom['tweet'] = ['id', 'tweet', 'user_id', 'username', 'place']

# Tweets desde el 1 de Abril hasta el 1 de Mayo
c.Since = '2020-04-01 00:00:00'
c.Until = '2020-05-01 00:00:00'

# Almacenar en CSV
c.Store_object = True

# Tweets con los hashtags
term1 = ['gobierno', 'estado', 'españa', 'gestion', 'autoridad', 'autoridades']
term2 = ['coronavirus', 'virus', 'covid', 'covid19', 'pandemia', 'crisis sanitaria']

# Buscar twets combinando términos de ambas listas
for t1 in term1:
    for t2 in term2:
        c.Search = [t1, t2]

        # Realizar búsqueda
        twint.run.Search(c)

# Almacenar los tweets de forma que no se repitan
aux = twint.output.tweets_list

tweets = dict()

for t in aux:
    if t.id not in tweets:
        tweets[t.id] = {'tweet': t.tweet, 'user_id': t.user_id,
                        'username': t.username, 'place': t.place}

# Almacenar los tweets en fichero csv
with open('tweets.csv', 'w') as f:
    # Añadir cabecera
    f.write(';'.join(c.Custom['tweet'])+'\n')

    # Añadir datos
    for tweet in tweets:
        f.write(str(tweet) + ';' +
                ';'.join('"' + v.replace(';', ',').replace('"', '\'').replace('\n','\t') +'"' if isinstance(v, str) else str(v) for v in tweets[tweet].values())+
                '\n')
