from neo4j import GraphDatabase

# TUS CREDENCIALES (Las mismas que en city_generator)
URI = "neo4j+ssc://5d9c9334.databases.neo4j.io"
AUTH = ("neo4j", "oTzaPYT99TgH-GM2APk0gcFlf9k16wrTcVOhtfmAyyA")

def nuke_database():
    print("☢️  INICIANDO PROTOCOLO DE LIMPIEZA TOTAL...")
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        # Borra todos los nodos y relaciones
        # Usamos apoc.periodic.iterate si fuera muy grande, 
        # pero para la versión free esto funciona bien:
        session.run("MATCH (n) DETACH DELETE n")
        
    driver.close()
    print("✨ Base de datos vacía. Lista para nueva simulación.")

if __name__ == "__main__":
    nuke_database()