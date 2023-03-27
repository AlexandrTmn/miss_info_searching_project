# -*- coding: utf-8 -*-
"""
Acquiring connection for database
"""
# imports
import edgedb

# global use of database
global conn


# Connecting to EdgeDB
def db_connection():
    host = 'localhost'
    port = 10700
    user = "edgedb"
    password = 'v38YIBf6YEUTWTSWQKAKcjG7'
    database = 'edgedb'

    cn = edgedb.create_client(
        host=host,
        port=port,
        user=user,
        database=database, tls_security='insecure', password=password
    )
    return cn
