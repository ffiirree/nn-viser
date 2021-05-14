<template>
    <el-container class="main-container">
        <el-header class="main-header">
            <div @click="collapseChanged" class="collapse-btn">
                <i v-if="!collapse" class="el-icon-s-fold"/>
                <i v-else class="el-icon-s-unfold"/>
            </div>
            <router-link to="/" class="title">VISUALIZER</router-link>
        </el-header>
        <el-container>
            <el-aside class="left-aside">
                <el-menu class="aside-menu"
                         :collapse="collapse"
                         :default-active="$route.path"
                         background-color="#333333"
                         text-color="#dddddd"
                         unique-opened
                         router>
                    <template v-for="item in menu">
                        <template v-if="item.sub_items">
                            <el-submenu :index="item.index" :key="item.index">
                                <template slot="title"><i /><v-icon class="menu-item-icon" :name="item.icon"/> <span>{{ item.title }}</span></template>
                                <template v-for="sub_item in item.sub_items">
                                    <el-menu-item class="sub-menu-item" :index="sub_item.index" :key="sub_item.index">
                                        <i class="dot" :class="{'activated':sub_item.index === $route.fullPath}"/>{{ sub_item.title }}
                                    </el-menu-item>
                                </template>
                            </el-submenu>
                        </template>
                        <template v-else>
                            <el-menu-item :index="item.index" :key="item.index"><v-icon class="menu-item-icon" :name="item.icon"/><span slot="title">{{ item.title }}</span></el-menu-item>
                        </template>
                    </template>
                </el-menu>
            </el-aside>
            <el-container class="content" :class="{ 'content-collapse':collapse }">
                <router-view/>
            </el-container>
        </el-container>
    </el-container>
</template>

<script>
    import Icon from "vue-awesome/components/Icon";

    export default {
        name: 'z-page',
        data() {
            return {
                user: {},
                collapse: false,
                notifications: [],

                tags: []
            }
        },
        props: [
            'menu',
        ],
        components: {
            'v-icon': Icon
        },
        methods: {
            collapseChanged() {
                this.collapse = !this.collapse;
            }
        },
        sockets: {
            connect() {
                console.log(`socket connected.`);
            },

            disconnect() {
                console.log("socket disconnected");
            },
        }
    }
</script>

<style rel="stylesheet/scss" lang="scss">
    .dot {
        background-color: #dddddd;
        display: inline-block;
        width: 3px;
        height: 3px;
        margin-right: 7px;
        border-radius: 50%;
    }

    .activated {
        background-color: #3a90f2;
    }

    .main-container {
        height: 100%;
        width: 100%;
        overflow: hidden;

        .main-header {
            width: 100%;
            height: 50px !important;
            padding: 0 !important;
            box-sizing: border-box;

            color: #ffffff;
            background-color: #262626;

            .collapse-btn {
                line-height: 50px;
                height: 50px;
                width: 65px;
                display: inline-block;
                text-align: center;
                cursor: pointer;
            }
            .collapse-btn:hover {
                background-color: #363646;
            }

            .title {
                display: inline-block;
                text-decoration: none;
                color: white;
                line-height: 50px;
                user-select: none;
                
                padding-left: 10px;
            }
        }

        .left-aside {
            position: absolute;
            top: 50px;
            bottom: 0;
            left: 0;
            user-select: none;

            overflow-y: scroll;
            ::-webkit-scrollbar {
                width: 0;
            }

            .aside-menu {
                height: 100%;
            }

            .menu-item-icon {
                width: 25px;
                text-align: center;
            }

            .aside-menu {
                border-right: 1px solid #333333;
            }

            .aside-menu:not(.el-menu--collapse) {
                width: 200px;
            }

            .sub-menu-item {
                background-color: #222222 !important;
            }

            .sub-menu-item:hover {
                background-color: #292929 !important;
            }
        }

        .content {
            position: absolute;
            top: 50px;
            left: 200px;
            bottom: 0;
            right: 0;
            -webkit-transition: left .3s ease-in-out;
            transition: left .3s ease-in-out;
            background-color: #f6f7f7;
            overflow-y: auto;

            .page {
                width: 100%;
                height: 100%;
                // display: block;
                // padding: 20px;
                box-sizing: border-box;
            }
        }

        .content-collapse {
            left: 65px;
        }
    }
</style>
