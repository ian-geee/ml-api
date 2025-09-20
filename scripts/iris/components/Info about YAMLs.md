YAMLs files in Azure ML are a way to define instances of Classes : CommandComponent, CommandJob, PipelineComponent, PipelineJob

<span style="color:red">Les détails ne sont pas importants, ce qui compte c'est de comprendre l'utilisation des yaml en tant qu'instances de classes définies au préalable</span>

### CommandComponent:

   from azure.ai.ml.entities import CommandComponent

   component = CommandComponent(
       name="sample_command_component_basic",
       display_name="CommandComponentBasic",
       description="This is the basic command component",
       tags={"tag": "tagvalue", "owner": "sdkteam"},
       version="1",
       outputs={"component_out_path": {"type": "uri_folder"}},
       command="echo Hello World",
       code="./src",
       environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:33",
   )

### CommandJob:

   command_job = CommandJob(
       code="./src",
       command="python train.py --ss {search_space.ss}",
       inputs={"input1": Input(path="trial.csv")},
       outputs={"default": Output(path="./foo")},
       compute="trial",
       environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:33",
       limits=CommandJobLimits(timeout=120),
   )

### PipelineJob:

 from azure.ai.ml.entities import PipelineJob, PipelineJobSettings

   pipeline_job = PipelineJob(
       description="test pipeline job",
       tags={},
       display_name="test display name",
       experiment_name="pipeline_job_samples",
       properties={},
       settings=PipelineJobSettings(force_rerun=True, default_compute="cpu-cluster"),
       jobs={"component1": component_func(component_in_number=1.0, component_in_path=uri_file_input)},
   )

   ml_client.jobs.create_or_update(pipeline_job)


