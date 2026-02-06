from src.llmconfig import models
from langchain_core.messages import HumanMessage
from ..prompt import *
from ..state import *
from langchain.agents import create_agent
from pydantic import BaseModel,Field
from langgraph.types import interrupt,Command
from src.llm_wrapper import llm_wrapper

#大模型选择
ds_V3 = models["DeepSeek_V3"]
ds_R1 = models["DeepSeek_R1"]

class defense_evidence_format(BaseModel):
    current_evidence: list[Evidence] | Evidence
    messages: str

async def defense_object_control(state: CourtState)-> Command:
    """
    【路由节点】发生在公诉人念完起诉书之后，在这个节点，要处理interrupt后用户传入参数defense_object_quit
    如果false则跳转到辩护代理人发表起诉书异议（即defense_objection节点），
    如果True则直接跳转到公诉人提问（pros_question)
    """
    user_object = interrupt("公诉人现在已经念完了起诉书，你是否有异议？(true/false)")
    if not user_object:
        return Command(goto= "pros_question")
    else:
        return Command(goto= "defense_objection")
    
async def defense_objection(state: CourtState)-> CourtState:
    """
    【动态人类节点】这个节点辩护代理人可以发表自己对于起诉书的异议，
    在当前模式中，辩护代理人为用户，于是需要中断，更新用户输入
    """
    user_input = interrupt("请发表对起诉书的异议：")
    response = HumanMessage(content=user_input,
                            name = f"辩护代理人{state.meta.attorney_name}")
    return{
        "messages": [response]
    }

async def defense_reply(state: CourtState)-> CourtState:
    """
    【动态AI节点】AI充当被告人接受控方和辩方的提问
    """
    defense_reply = await DEFENSE_REPLY.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_reply
    )
    reply = response.get("messages",[])[-1]
    reply.name = f"被告人{state.meta.defendant_name}"
    return {
        "messages": [reply]
    }

def reply_control(state: CourtState)-> str:
    """
    【路由函数】检查状态参数中的pros_question_round,defense_question_quit,
    决定在被告人发言结束后由哪一方发言
    """
    if state.pros_question_rounds == -1:
        return "defense_question"
    else:
        return "pros_question"
    
async def defense_question_control(state: CourtState)-> Command:
    """
    【路由节点】接受用户输入的的参数，判断用户是否还有意愿继续提问
    """
    user_question_desire = interrupt("你是否还需要向被告人提问？（true/false）")
    if not user_question_desire == True:   # 辩护代理人放弃提问
        return Command(goto= "pros_summary",
                       update= {
                           "messages": HumanMessage(content="合议庭，辩护人的提问到此结束。",
                                                    name = f"辩护代理人{state.meta.attorney_name}")
                       })
    else:
        return Command(goto= "defense_question")
    
async def defense_question(state: CourtState)-> CourtState:
    """
    【动态人类节点】中断后消费临时参数：user_input
    即用户提问/放弃提问的语句（如果放弃提问，记得提示用户输入放弃相关的语句）
    """
    user_input = interrupt("请输入你想提问的问题：")
    response = HumanMessage(content=user_input,
                            name = f"辩护代理人{state.meta.attorney_name}")
    return{
        "messages": [response]
    }

async def defense_summary(state: CourtState)-> CourtState:
    """
    【动态人类节点】中断后消费临时参数：user_input
    用户需上传问被告小结
    """
    user_input = interrupt("请输入你的问被告小结：")
    response = HumanMessage(content=user_input,
                            name = f"辩护代理人{state.meta.attorney_name}")
    return{
        "messages": [response]
    }

async def defense_cross(state: CourtState)-> CourtState:
    """
    【动态人类节点】中断后消费临时参数：user_input
    用户需上传对该证据的质证
    """
    user_input = interrupt("请输入你的质证意见：")
    response = HumanMessage(content=user_input,
                            name = f"辩护代理人{state.meta.attorney_name}")
    return{
        "messages": [response]
    }
 
async def defense_evidence_control(state: CourtState)-> Command:
    """
    【路由节点】在审判长询问辩护人是否有补充质证意见的时候，辩护人需要输入选择：
    true为有补充质证意见，false为没有补充质证意见。
    """
    user_evidence_desire = interrupt("你是否有补充证据要提出？（true/false）")
    if user_evidence_desire == True:   # 辩护代理人有补充证据
        return Command(goto= "defense_show_evidence")
    else:
        return Command(goto= "judge_start_debate",
                       update= {
                           "messages": HumanMessage(content= "合议庭，辩护人没有要补充的证据了。",
                                                    name = f"辩护代理人{state.meta.attorney_name}")
                       })
    
async def defense_show_evidence(state: CourtState) -> CourtState:
    """
    【动态人类节点】辩护人提出证据内容并且做简要说明
    """
    # 1. 获取原始输入（这是一个 dict）
    raw_input = interrupt("请输入补充证据以及证据意见（注意JSON格式规范）")
    
    # 2. 【关键步骤】将字典转换为 Pydantic 模型对象
    # 这样既能进行数据校验，又能启用点号访问 (.messages)
    try:
        user_input = defense_evidence_format.model_validate(raw_input)
    except Exception as e:
        # 建议加上简单的错误处理，防止用户输入的 JSON 格式对不上
        raise ValueError(f"输入格式错误，需要符合 defense_evidence_format 定义: {e}")

    # 3. 现在可以使用点号访问属性了
    response = HumanMessage(
        content=user_input.messages,
        name=f"辩护代理人{state.meta.attorney_name}"
    )
    
    return {
        "messages": [response],
        "current_evidence": user_input.current_evidence
    }

async def defense_self_statement(state: CourtState)-> CourtState:
    """
    【动态AI节点】被告人自行辩护
    """
    defense_self_statement = await DEFENSE_SELF_STATEMENT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model= ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_self_statement
    )
    statement = response.get("messages",[])[-1]
    statement.name = f"被告人{state.meta.defendant_name}"
    return {
        "messages": [statement]
    }

async def defense_statement(state: CourtState)-> CourtState:
    """
    【动态人类节点】辩护人进行辩护意见的发表
    """
    user_input = interrupt("请输入你的第一轮辩护意见：")
    response = HumanMessage(content= user_input,
                            name= f"辩护代理人{state.meta.attorney_name}")
    return {
        "messages": [response]
    }

async def defense_focus(state: CourtState)-> CourtState:
    """
    【动态人类节点】辩护人对争议焦点的回应
    """
    user_input = interrupt("请输入你针对当前争议焦点的回应：")
    response = HumanMessage(content= user_input,
                            name= f"辩护代理人{state.meta.attorney_name}")
    return {
        "messages": [response],
        "pros_focus_rounds": state.pros_focus_rounds - 1
    }

async def defense_sumup(state: CourtState)-> CourtState:
    """
    【动态人类节点】辩护人作总结的辩护意见发表
    """
    user_input = interrupt("请输入你的总结的辩护意见：")
    response = HumanMessage(content= user_input,
                            name= f"辩护代理人{state.meta.attorney_name}")
    return {
        "messages": [response]
    }

async def defense_final_statement(state: CourtState)-> CourtState:
    """
    【动态AI节点】由被告人作最终陈述
    """
    defense_final_statement = await DEFENSE_FINAL_STATEMENT.ainvoke(
        {
            "case_info" : state.meta,
            "messages" : state.messages
        }
    )
    defendant = create_agent(model=ds_V3)
    response = await llm_wrapper.ainvoke_with_retry(
        defendant.ainvoke,
        defense_final_statement
    )
    sumup = response.get("messages",[])[-1]
    sumup.name = f"被告人{state.meta.defendant_name}"
    return {
        "messages" : [sumup]
    }
