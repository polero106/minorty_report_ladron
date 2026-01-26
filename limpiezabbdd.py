from neo4j import GraphDatabase

# TUS CREDENCIALES (Las mismas que en city_generator)
URI = "neo4j+ssc://c6226feb.databases.neo4j.io"
AUTH = ("neo4j", "8G7YN9W2V7Y_RQDCqWTHrryWd-G8GnNIF3ep9vslp6k")

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