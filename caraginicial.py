import json
from neo4j import GraphDatabase

# Configuración (Ajusta según tu entorno local o AuraDB)
URI = "neo4j+ssc://5d9c9334.databases.neo4j.io"
AUTH = ("neo4j", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA")


def cargar_datos(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    driver = GraphDatabase.driver(URI, auth=AUTH)

    query_personas = """
    UNWIND $batch as item
    MERGE (p:Persona {id: item.id})
    SET p.nombre = item.nombre, p.historial = item.historial, p.embedding = [0.1, 0.5] // Dummy embedding
    """

    query_ubicaciones = """
    UNWIND $batch as item
    MERGE (u:Ubicacion {id: item.id})
    SET u.distrito = item.distrito, u.peligro = item.peligro
    """

    query_crimenes = """
    UNWIND $batch as item
    MERGE (c:Crime {id: item.id})
    SET c.tipo = item.tipo, c.timestamp = item.timestamp
    """

    query_relaciones = """
    UNWIND $batch as item
    MATCH (s {id: item.source})
    MATCH (t {id: item.target})
    CALL apoc.create.relationship(s, item.tipo, {}, t) YIELD rel
    RETURN count(rel)
    """

    with driver.session() as session:
        print("Cargando Personas...")
        session.run(query_personas, batch=data['personas'])
        print("Cargando Ubicaciones...")
        session.run(query_ubicaciones, batch=data['ubicaciones'])
        print("Cargando Crímenes...")
        session.run(query_crimenes, batch=data['crimenes_futuros'])
        print("Creando Relaciones...")
        session.run(query_relaciones, batch=data['relaciones'])

    driver.close()
    print("¡Datos cargados! La 'Policía' ya tiene un grafo para analizar.")


# Ejecutar
if __name__ == "__main__":
    cargar_datos('datos/prueba.json')
