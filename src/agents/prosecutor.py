from src.llmconfig import models
from langchain_core.messages import AIMessage
from ..prompt import *
from ..state import *
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy,ToolStrategy
from pydantic import BaseModel,Field
from langgraph.types import Command
from src.llm_wrapper import llm_wrapper

class evidence_response_format(BaseModel):
    current_evidence : List[Evidence] | None = Field(description= "这里的取值有三种情况，分别对应三种举证模式")
    evidence_show_type : Evidence_Show_Enum = Field(description= "这是一个枚举类型，只有三种字符串取值，分别是“independent”，“union”,和“quit”")


# 大模型选择
ds_V3 = models["DeepSeek_V3"]
ds_R1 = models["DeepSeek_R1"]
kimi_k2_5 = models["KIMI_K2.5"]


async def pros_indictment(state: CourtState)-> CourtState:
    """
    【静态节点】公诉人宣读起诉书
    """
    return {
        "phase" : PhaseEnum.INVESTIGATION,
        "messages": [AIMessage(content=state.meta.statement_charge,
                                 name = f"公诉人{state.meta.prosecutor_name}")]
    }

def pros_round_control(state: CourtState) -> str:
    """
    【路由函数】通过state的一个临时参数pros_question_rounds
    来控制提问人是谁
    """
    if state.pros_question_rounds >=0:
        return "defense_reply"
    else:
        return "defense_question"

async def pros_question(state: CourtState)-> CourtState:
    """
    【动态AI节点】如果还有提问次数，公诉人向被告针对案件事实向被告人进行提问；如果没有，则进入问被告小结。
    """
    # 先判断当前剩余提问次数
    if state.pros_question_rounds > 0:
        rounds = state.pros_question_rounds - 1 # 扣除一次提问次数
        pros_question = await PROS_QUESTION.ainvoke(
            {
                "case_info" : state.meta,
                "messages" : state.messages
            }
        )
        prosecutor = create_agent(model=ds_V3)
        response = await llm_wrapper.ainvoke_with_retry(
            prosecutor.ainvoke,
            pros_question
        )
        question = response.get("messages",[])[-1]
        question.name = f"公诉人{state.meta.prosecutor_name}"
        return {
            "messages" : [question],
            "pros_question_rounds" : rounds
        }
    else:
        return {
            "messages" : [AIMessage(content= "合议庭，公诉人对被告人的提问到此结束。",
                                    name= f"公诉人{state.meta.prosecutor_name}")],
            "pros_question_rounds" : -1   # 返回一个-1，以便条件边（>=0)成功路由
        }

async def pros_summary(state: CourtState)-> CourtState:
    """【动态AI节点】公诉人根据双方提问进行问被人小结"""
    pros_summary = await PROS_SUMMARY.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    prosecutor = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        prosecutor.ainvoke,
        pros_summary
    )
    summary = response.get("messages",[])[-1]
    summary.name = f"公诉人{state.meta.prosecutor_name}"
    return {
        "messages" : [summary]
    }

async def pros_evidence_decision(state: CourtState)-> Command:
    """
    【路由节点】公诉人根据场上情形，在正式举证之前要先做一步判断：
    是应当单一证据举证、还是联合举证、还是放弃举证？
    放弃举证的话直接跳转到审判长询问补充质证意见，进行举证则跳转到公诉人宣读证据
    """
    evidence_decision = await PROS_EVIDENCE_DECISION.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "evidence_list" : state.evidence_list
        }
    )
    prosecutor = create_agent(model=kimi_k2_5,response_format=ToolStrategy(evidence_response_format))
    response = await llm_wrapper.ainvoke_with_retry(
        prosecutor.ainvoke,
        evidence_decision
    )
    decision: evidence_response_format = response["structured_response"]
    if decision.evidence_show_type == "quit" or state.pros_evidence_rounds <= 0:
        return Command(update={
            "current_evidence": decision.current_evidence,
            "evidence_show_type": decision.evidence_show_type,
            "messages": AIMessage(content= "感谢合议庭，公诉人的举证到此结束。",
                                  name= f"公诉人{state.meta.prosecutor_name}")
        },
        goto= "judge_confirm")
    else:
        return Command(
            update={
            "current_evidence": decision.current_evidence,
            "evidence_show_type": decision.evidence_show_type  
            },
            goto= "pros_show_evidence"
        )
    
async def pros_show_evidence(state: CourtState)-> CourtState:
    """
    【动态AI节点】公诉人根据前序的决定，宣读证据并作说明，后续由辩护人质证
    """
    show_evidence = await PROS_SHOW_EVIDENCE.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "current_evidence": state.current_evidence
        }
    )
    prosecutor = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        prosecutor.ainvoke,
        show_evidence
    )
    show = response.get("messages",[])[-1]
    show.name = f"公诉人{state.meta.prosecutor_name}"
    rounds = state.pros_evidence_rounds - 1
    return {
        "messages": [show],
        "pros_evidence_rounds": rounds
    }

async def pros_cross(state: CourtState)-> CourtState:
    """
    【动态AI节点】公诉人对辩护人提出的补充证据做回应。
    """
    pros_cross = await PROS_CROSS.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages,
            "current_evidence": state.current_evidence
        }
    )
    prosecutor = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        prosecutor.ainvoke,
        pros_cross
    )
    cross = response.get("messages",[])[-1]
    cross.name = f"公诉人{state.meta.prosecutor_name}"
    return {
        "messages": [cross]
    }

async def pros_statement(state: CourtState)-> CourtState:
    """
    【动态AI节点】公诉人发表第一轮公诉意见
    """
    pros_statement = await PROS_STATEMENT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    prosecutor = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        prosecutor.ainvoke,
        pros_statement
    )
    statement = response.get("messages",[])[-1]
    statement.name = f"公诉人{state.meta.prosecutor_name}"
    return {
        "messages" : [statement]
    }

async def pros_focus(state: CourtState)-> Command:
    """
    【路由AI节点】公诉人针对当前争议焦点进行公诉意见发表,节点一开始需要检查是否达到争议焦点轮数上限
    """
    if state.pros_focus_rounds > 0:
        current_focus = state.focus[state.focus_index]
        pros_focus = await PROS_FOCUS.ainvoke(
            {
                "case_info" : state.meta,
                "messages" : state.messages,
                "current_focus": current_focus 
            }
        )
        prosecutor = create_agent(model=ds_V3)
        response = await llm_wrapper.ainvoke_with_retry(
            prosecutor.ainvoke,
            pros_focus
        )
        focus = response.get("messages",[])[-1]
        focus.name = f"公诉人{state.meta.prosecutor_name}"
        return Command(
            update={
                "messages": [focus]
            },
            goto="defense_focus")
    else:
        return Command(update={
            "focus_index": state.focus_index + 1
        },
        goto="focus")
    
async def pros_sumup(state: CourtState)-> CourtState:
    """
    【动态AI节点】公诉人发表总结的公诉意见
    """
    pros_sumup = await PROS_SUMUP.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    prosecutor = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        prosecutor.ainvoke,
        pros_sumup
    )
    sumup = response.get("messages",[])[-1]
    sumup.name = f"公诉人{state.meta.prosecutor_name}"
    return {
        "messages" : [sumup]
    }