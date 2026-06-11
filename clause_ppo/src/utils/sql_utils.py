"""
reconstruct_sql: Spider parsed sql dict -> executable SQL string.

Design notes:
- Purpose is execution-oracle verification only, NOT pretty-printing.
  The output may differ syntactically from the gold SQL but must be semantically
  equivalent for the purpose of detecting result changes.
- JOINs are reconstructed as comma-separated tables with join conditions
  merged into the WHERE clause. SQLite handles this correctly.
- Column references are always table-qualified when multiple tables are in FROM,
  preventing ambiguity errors.
- Limitations: INTERSECT/UNION/EXCEPT set operations are appended as-is.
  Subqueries in FROM (type "sql") are wrapped as derived tables.
"""

from typing import Any, Optional


# ── Operator lookup tables (Spider spec) ────────────────────────────────────

AGG_OPS   = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
UNIT_OPS  = ['', '-', '+', '*', '/']
WHERE_OPS = ['NOT', 'BETWEEN', '=', '>', '<', '>=', '<=', '!=',
             'IN', 'LIKE', 'IS', 'EXISTS']


# ── Column / table name helpers ──────────────────────────────────────────────

def _col_name(col_id: int, tables: dict) -> str:
    """Return the raw column name for a col_id (0 → '*')."""
    if col_id == 0:
        return '*'
    return tables['column_names_original'][col_id][1]


def _table_idx_of_col(col_id: int, tables: dict) -> int:
    """Return the table_idx that owns col_id (-1 for the wildcard col)."""
    return tables['column_names_original'][col_id][0]


def _table_name(table_idx: int, tables: dict) -> str:
    return tables['table_names_original'][table_idx]


# ── Low-level unit converters ────────────────────────────────────────────────

def _col_unit_to_sql(col_unit: list, tables: dict,
                     qualify: bool = False,
                     table_idx_map: Optional[dict] = None) -> str:
    """
    col_unit = [agg_id, col_id, is_distinct]
    qualify=True → prefix column with its table name (needed for multi-table queries).
    """
    agg_id, col_id, is_distinct = col_unit
    col = _col_name(col_id, tables)

    if qualify and col != '*' and table_idx_map is not None:
        t_idx = _table_idx_of_col(col_id, tables)
        if t_idx in table_idx_map:
            col = f"{table_idx_map[t_idx]}.{col}"

    distinct = 'DISTINCT ' if is_distinct else ''
    if agg_id > 0:
        return f"{AGG_OPS[agg_id]}({distinct}{col})"
    return f"{distinct}{col}"


def _val_unit_to_sql(val_unit: list, tables: dict,
                     qualify: bool = False,
                     table_idx_map: Optional[dict] = None) -> str:
    """val_unit = [unit_op, col_unit1, col_unit2_or_null]"""
    unit_op, col_unit1, col_unit2 = val_unit
    s1 = _col_unit_to_sql(col_unit1, tables, qualify, table_idx_map)
    if col_unit2 is not None and unit_op > 0:
        s2 = _col_unit_to_sql(col_unit2, tables, qualify, table_idx_map)
        return f"({s1} {UNIT_OPS[unit_op]} {s2})"
    return s1


def _val_to_sql(val: Any, tables: dict,
                qualify: bool = False,
                table_idx_map: Optional[dict] = None) -> str:
    """
    Convert a condition value (val1 or val2) to a SQL fragment.
    val can be: None | number | str | col_unit (list[3]) | sql_dict (dict)
    """
    if val is None:
        return 'NULL'
    if isinstance(val, dict):
        # Nested subquery
        return f"({reconstruct_sql(val, tables)})"
    if (isinstance(val, list) and len(val) == 3
            and isinstance(val[0], int) and isinstance(val[1], int)):
        # col_unit: [agg_id, col_id, is_distinct]
        return _col_unit_to_sql(val, tables, qualify, table_idx_map)
    if isinstance(val, str):
        # Spider stores string literals with surrounding double-quotes, e.g. '"Alabama"'.
        # Strip them and re-wrap in single quotes for SQLite compatibility.
        if val.startswith('"') and val.endswith('"') and len(val) >= 2:
            inner = val[1:-1]
        else:
            inner = val
        escaped = inner.replace("'", "''")
        return f"'{escaped}'"
    return str(val)


def _cond_to_sql(cond: list, tables: dict,
                 qualify: bool = False,
                 table_idx_map: Optional[dict] = None) -> str:
    """
    cond = [not_flag, op_id, val_unit, val1, val2]
    Produces the SQL fragment for one condition.
    """
    not_flag, op_id, val_unit, val1, val2 = cond
    col_str  = _val_unit_to_sql(val_unit, tables, qualify, table_idx_map)
    op_str   = WHERE_OPS[op_id]
    not_str  = 'NOT ' if not_flag else ''

    if op_id == 1:   # BETWEEN
        v1 = _val_to_sql(val1, tables, qualify, table_idx_map)
        v2 = _val_to_sql(val2, tables, qualify, table_idx_map)
        return f"{col_str} {not_str}BETWEEN {v1} AND {v2}"
    if op_id == 8:   # IN
        v1 = _val_to_sql(val1, tables, qualify, table_idx_map)
        return f"{col_str} {not_str}IN ({v1})"
    if op_id == 10:  # IS (NULL)
        return f"{col_str} IS {not_str}NULL"
    if op_id == 11:  # EXISTS
        v1 = _val_to_sql(val1, tables, qualify, table_idx_map)
        return f"{not_str}EXISTS {v1}"

    v1 = _val_to_sql(val1, tables, qualify, table_idx_map)
    return f"{col_str} {not_str}{op_str} {v1}"


def _cond_list_to_sql(cond_list: list, tables: dict,
                      qualify: bool = False,
                      table_idx_map: Optional[dict] = None) -> str:
    """Convert a condition list (interleaved with 'and'/'or' strings) to SQL."""
    parts = []
    for item in cond_list:
        if isinstance(item, str):   # 'and' / 'or'
            parts.append(item.upper())
        else:
            parts.append(_cond_to_sql(item, tables, qualify, table_idx_map))
    return ' '.join(parts)


# ── Main reconstruction function ─────────────────────────────────────────────

def reconstruct_sql(sql_dict: dict, tables: dict) -> str:
    """
    Convert a Spider parsed `sql` dict back into an executable SQL string.

    The output is not guaranteed to match the gold SQL string token-for-token,
    but it will be semantically equivalent for the six main clauses (SELECT,
    FROM, WHERE, GROUP BY, HAVING, ORDER BY) plus LIMIT and set operations.

    Args:
        sql_dict: The 'sql' field from a Spider dataset entry.
        tables:   The tables.json entry for the same db_id.

    Returns:
        Executable SQL string.
    """
    parts: list[str] = []

    # ── FROM ─────────────────────────────────────────────────────────────────
    from_clause  = sql_dict.get('from') or {}
    table_units  = from_clause.get('table_units', [])
    join_conds   = from_clause.get('conds', [])

    # Build table_idx → table_name map for column qualification
    table_idx_map: dict[int, str] = {}
    from_strs: list[str] = []
    sub_idx = 0
    for tu_type, tu_val in table_units:
        if tu_type == 'table_unit':
            name = _table_name(tu_val, tables)
            table_idx_map[tu_val] = name
            from_strs.append(name)
        else:
            # Subquery in FROM
            alias = f"sub{sub_idx}"
            sub_idx += 1
            from_strs.append(f"({reconstruct_sql(tu_val, tables)}) AS {alias}")

    qualify = len(table_idx_map) > 1
    parts.append(f"FROM {', '.join(from_strs)}")

    # ── SELECT ───────────────────────────────────────────────────────────────
    select      = sql_dict.get('select') or [False, []]
    is_distinct = select[0]
    agg_items   = select[1] if len(select) > 1 else []

    select_cols: list[str] = []
    for agg_id, val_unit in agg_items:
        col_str = _val_unit_to_sql(val_unit, tables, qualify, table_idx_map)
        # agg_id at the select level wraps the entire val_unit expression
        if agg_id > 0:
            col_str = f"{AGG_OPS[agg_id]}({col_str})"
        select_cols.append(col_str)

    distinct_str = 'DISTINCT ' if is_distinct else ''
    # Prepend SELECT so it appears before FROM in the final join
    parts.insert(0, f"SELECT {distinct_str}{', '.join(select_cols)}")

    # ── WHERE (merge join conds + explicit WHERE) ────────────────────────────
    where = sql_dict.get('where') or []
    all_where: list = []
    if join_conds:
        all_where.extend(join_conds)
    if where:
        if all_where:
            all_where.append('and')
        all_where.extend(where)

    if all_where:
        parts.append(f"WHERE {_cond_list_to_sql(all_where, tables, qualify, table_idx_map)}")

    # ── GROUP BY ─────────────────────────────────────────────────────────────
    group_by = sql_dict.get('groupBy') or []
    if group_by:
        gb_cols = [_col_unit_to_sql(cu, tables, qualify, table_idx_map)
                   for cu in group_by]
        parts.append(f"GROUP BY {', '.join(gb_cols)}")

    # ── HAVING ───────────────────────────────────────────────────────────────
    having = sql_dict.get('having') or []
    if having:
        parts.append(f"HAVING {_cond_list_to_sql(having, tables, qualify, table_idx_map)}")

    # ── ORDER BY ─────────────────────────────────────────────────────────────
    order_by = sql_dict.get('orderBy') or []
    if order_by and len(order_by) == 2:
        direction, val_units = order_by
        ob_cols = [_val_unit_to_sql(vu, tables, qualify, table_idx_map)
                   for vu in val_units]
        if ob_cols:
            parts.append(f"ORDER BY {', '.join(ob_cols)} {direction.upper()}")

    # ── LIMIT ────────────────────────────────────────────────────────────────
    limit = sql_dict.get('limit')
    if limit is not None:
        parts.append(f"LIMIT {int(limit)}")

    result = ' '.join(parts)

    # ── Set operations ────────────────────────────────────────────────────────
    for op in ('intersect', 'union', 'except'):
        nested = sql_dict.get(op)
        if nested:
            result += f" {op.upper()} {reconstruct_sql(nested, tables)}"

    return result


def extract_sql_from_text(text: str) -> str:
    """
    Extract SQL from model-generated text that may contain markdown or formatting.
    
    Args:
        text: Raw text potentially containing SQL
        
    Returns:
        Clean SQL string
    """
    # Remove common markdown patterns
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith('```sql') and text.endswith('```'):
        text = text[6:-3].strip()
    elif text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
    
    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        'Answer:', 'SQL:', 'Query:', '[SQL]', '<sql>', 'The SQL query is:',
        'Here is the SQL:', 'The complete query is:'
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Remove common suffixes
    suffixes_to_remove = ['</sql>', '[/SQL]', '</answer>']
    for suffix in suffixes_to_remove:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()
    
    # Take only the first line if multiple lines (often the SQL is on the first line)
    lines = text.split('\n')
    if lines:
        first_line = lines[0].strip()
        # If first line looks like SQL, use it
        if first_line and any(keyword in first_line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']):
            text = first_line
    
    # Remove trailing periods or semicolons
    text = text.rstrip('.;')
    
    return text.strip()
