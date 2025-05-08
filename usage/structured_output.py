#https://python.langchain.com/docs/concepts/structured_outputs/
#https://python.langchain.com/docs/how_to/structured_output/#the-with_structured_output-method
# Define schema
schema = {"foo": "bar"}
# Bind schema to model
model_with_structure = model.with_structured_output(schema)
# Invoke the model to produce structured output that matches the schema
structured_output = model_with_structure.invoke(user_input)